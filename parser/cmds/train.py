# -*- coding: utf-8 -*-
from datetime import datetime, timedelta

from parser.cmds.cmd import CMD
from parser.helper.metric import LikelihoodMetric, Metric
from parser.helper.loader_wrapper import DataPrefetcher
import torch
from torch.optim import lr_scheduler

# from parser.helper.util import *
from parser.helper.data_module import DataModule

from torch.utils.tensorboard import SummaryWriter
import wandb
import multiprocessing as mp

from pathlib import Path
import math
import pickle
from utils import tensor_to_heatmap, span_to_tree_with_sent

import torch_support.reproducibility as reproducibility
from torch_support.train_support import get_logger
from torch_support.load_model import (
    set_model_dir,
    get_model_args,
    get_optimizer_args,
)
import torch_support.metric as metric
from parser.cmds.log import log_weight_histogram, log_rule_prob


class Train(CMD):
    def setup_logger(self):
        # Setup logger
        args = self.args
        console_level = args.get("console_level", "INFO")
        self.log = get_logger(args, console_level=console_level)
        self.log.info(f"Seed: {args.seed}")
        self.log.info("Create the model")
        self.log.info(f"{self.model}\n")
        self.log.info(self.optimizer)

        # Setup tensorboard writer
        if getattr(args, "tensorboard", False):
            self.writer = SummaryWriter(args.save_dir)

        # Setup WandB
        # start a new wandb run to track this script
        if getattr(args, "wandb", False):
            self.run = wandb.init(
                # set the wandb project where this run will be logged
                project="neural-grammar-induction",
                # track hyperparameters and run metadata
                config=args,
            )
            wandb.define_metric("train/step")
            wandb.define_metric("valid/epoch")
            wandb.define_metric("train/*", step_metric="train/step")
            wandb.define_metric("valid/*", step_metric="valid/epoch")

    def log_per_step(
        self, dev_f1_metric, dev_ll, dev_left_metric, dev_right_metric
    ):
        pass

    def log_per_epoch(
        self, dev_f1_metric, dev_ll, dev_left_metric, dev_right_metric
    ):
        # F1 score for each epoch
        tag = "valid"

        unary_jsd = metric.pairwise_js_div(self.model.rules["unary"])
        metric_list = {
            f"{tag}/epoch": self.epoch,
            f"{tag}/avg_likelihood": dev_ll.score,
            f"{tag}/perplexity": dev_ll.perplexity,
            f"{tag}/f1": dev_f1_metric.sentence_uf1,
            f"{tag}/exact": dev_f1_metric.sentence_ex,
            f"{tag}/unary_local_ppl": metric.local_ppl(
                self.model.rules["unary"]
            ),
            f"{tag}/unary_global_ppl": metric.global_ppl(
                self.model.rules["unary"]
            ),
            f"{tag}/unary_jsd_arithmetic": unary_jsd.mean(),
            f"{tag}/unary_jsd_geometric": metric.geometric_mean(unary_jsd),
        }

        if "rule" in self.model.rules:
            binary_jsd = metric.pairwise_js_div(
                self.model.rules["rule"].flatten(1)
            )
            metric_list.update(
                {
                    f"{tag}/binary_local_ppl": metric.local_ppl(
                        self.model.rules["rule"].flatten(1)
                    ),
                    f"{tag}/binary_global_ppl": metric.global_ppl(
                        self.model.rules["rule"].flatten(1)
                    ),
                    f"{tag}/binary_jsd_arithmetic": binary_jsd.mean(),
                    f"{tag}/binary_jsd_geometric": metric.geometric_mean(
                        binary_jsd
                    ),
                }
            )
        elif "head" in self.model.rules:
            head_jsd = metric.pairwise_js_div(self.model.rules["head"])
            left_jsd = metric.pairwise_js_div(self.model.rules["left"].T)
            right_jsd = metric.pairwise_js_div(self.model.rules["right"].T)
            metric_list.update(
                {
                    f"{tag}/head_local_ppl": metric.local_ppl(
                        self.model.rules["head"]
                    ),
                    f"{tag}/head_global_ppl": metric.global_ppl(
                        self.model.rules["head"]
                    ),
                    f"{tag}/head_jsd_arithmetic": head_jsd.mean(),
                    f"{tag}/head_jsd_geometric": metric.geometric_mean(
                        head_jsd
                    ),
                    f"{tag}/left_local_ppl": metric.local_ppl(
                        self.model.rules["left"].T
                    ),
                    f"{tag}/left_global_ppl": metric.global_ppl(
                        self.model.rules["left"].T
                    ),
                    f"{tag}/left_jsd_arithmetic": left_jsd.mean(),
                    f"{tag}/left_jsd_geometric": metric.geometric_mean(
                        left_jsd
                    ),
                    f"{tag}/right_local_ppl": metric.local_ppl(
                        self.model.rules["right"].T
                    ),
                    f"{tag}/right_global_ppl": metric.global_ppl(
                        self.model.rules["right"].T
                    ),
                    f"{tag}/right_jsd_arithmetic": right_jsd.mean(),
                    f"{tag}/right_jsd_geometric": metric.geometric_mean(
                        right_jsd
                    ),
                }
            )

        if self.left_binarization:
            metric_list.update(
                {
                    f"{tag}/f1_left": dev_left_metric.sentence_uf1,
                    f"{tag}/exact_left": dev_left_metric.sentence_ex,
                }
            )
        if self.right_binarization:
            metric_list.update(
                {
                    f"{tag}/f1_right": dev_right_metric.sentence_uf1,
                    f"{tag}/exact_right": dev_right_metric.sentence_ex,
                }
            )

        metric_dict = {
            f"{tag}/f1_length": dev_f1_metric.sentence_uf1_l,
            f"{tag}/Ex_length": dev_f1_metric.sentence_ex_l,
            f"{tag}/f1_left_length": dev_left_metric.sentence_uf1_l,
            f"{tag}/Ex_left_length": dev_left_metric.sentence_ex_l,
            f"{tag}/f1_right_length": dev_right_metric.sentence_uf1_l,
            f"{tag}/Ex_right_length": dev_right_metric.sentence_ex_l,
        }

        if getattr(self.args, "tensorboard", False):
            for k, v in metric_list.items():
                self.writer.add_scalar(f"{tag}/{k}", v, self.epoch)

            for k, v in metric_dict.items():
                for i, val in v.items():
                    self.writer.add_scalar(f"{tag}/{k}", val, i)

            # distribution of estimated span depth
            self.estimated_depth = dict(sorted(self.estimated_depth.items()))
            for k, v in self.estimated_depth.items():
                self.writer.add_scalar(
                    f"{tag}/estimated_depth", v / dev_f1_metric.n, k
                )

            # Model weight norm
            if getattr(self.args.train, "vector_histogram", False):
                log_weight_histogram(self.writer, self.model, self.epoch)
            # Rule probability distribution projection
            if getattr(self.args.train, "rule_embeddings", False):
                log_rule_prob(self.writer, self.model, self.epoch)

        if getattr(self.args, "wandb", False):
            self.run.log(metric_list)

    def setup_warmup(self, train_arg):
        train_arg.warmup_iter = int(
            train_arg.total_iter * train_arg.dambda_warmup
        )
        train_arg.warmup_start = int(
            train_arg.total_iter * (train_arg.dambda_warmup - 0.1)
        )
        train_arg.warmup_end = int(
            train_arg.total_iter * (train_arg.dambda_warmup + 0.1)
        )
        self.log.info(
            f"warmup start: {train_arg.warmup_start}, middle: {train_arg.warmup_iter}, end: {train_arg.warmup_end}"
        )
        return train_arg

    def __call__(self, args):
        self.args = args
        self.device = args.device

        # Load pretrained model
        if hasattr(args, "pretrained_model"):
            checkpoint = reproducibility.load(args.pretrained_model)
            # Load meta data
            start_epoch = checkpoint["epoch"] + 1
            # Load random state
            worker_init_fn = checkpoint["worker_init_fn"]
            generator = checkpoint["generator"]
        else:
            checkpoint = {"model": None, "optimizer": None}
            start_epoch = 1
            if not hasattr(args, "seed"):
                args.seed = int(str(torch.initial_seed())[:8])
            worker_init_fn, generator = reproducibility.fix_seed(args.seed)

        # Load dataset
        dataset = DataModule(
            args,
            generator=generator,
            worker_init_fn=worker_init_fn,
        )
        with open(f"{args.save_dir}/word_vocab.pkl", "wb") as f:
            pickle.dump(dataset.word_vocab, f)
        # Token setup
        self.pad_token = dataset.word_vocab.word2idx["<pad>"]
        self.unk_token = dataset.word_vocab.word2idx["<unk>"]
        self.mask_token = dataset.word_vocab.word2idx.get("<mask>", None)
        # Update vocab size & Add three different unknown tokens
        args.model.update({"V": len(dataset.word_vocab)})

        # Setup model
        set_model_dir("parser.model")
        self.model = get_model_args(
            args.model, self.device, checkpoint["model"]
        )
        if hasattr(args, "pretrained_terms"):
            self.model.load_state_dict(
                torch.load(
                    args.pretrained_terms,
                    map_location=self.device,
                )["model"],
                module="terms",
                frozen=True,
            )

        # Setup optimizer
        self.optimizer = get_optimizer_args(
            args.optimizer, self.model.parameters(), checkpoint["optimizer"]
        )
        if hasattr(self.args, "scheduler"):
            self.lr_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.args.scheduler.T_0,
                T_mult=self.args.scheduler.T_mult,
                eta_min=self.args.scheduler.eta_min,
            )

        # Setup logger
        self.setup_logger()

        total_time = timedelta()
        best_e, best_metric = 1, Metric()

        """
        Training
        """
        train_arg = getattr(args, "train")
        test_arg = getattr(args, "test")
        self.train_arg = train_arg
        self.test_arg = test_arg

        # Setup Validation
        eval_max_len = getattr(test_arg, "max_len", None)
        eval_loader = dataset.val_dataloader(max_len=eval_max_len)
        # Arguments for validation
        eval_depth = getattr(test_arg, "eval_depth", False)
        self.left_binarization = getattr(test_arg, "left_binarization", False)
        self.right_binarization = getattr(
            test_arg, "right_binarization", False
        )

        # iteration setup
        self.num_batch = len(
            dataset.train_dataloader(max_len=train_arg.max_len)
        )
        if hasattr(train_arg, "total_iter"):
            train_arg.max_epoch = math.ceil(
                train_arg.total_iter / self.num_batch
            )
            self.log.info(
                f"num of batch: {self.num_batch}, max epoch: {train_arg.max_epoch}"
            )
        train_arg.total_iter = train_arg.max_epoch * self.num_batch
        self.log.info(f"total iter: {train_arg.total_iter}")

        if getattr(train_arg, "dambda_warmup", False):
            train_arg = self.setup_warmup(train_arg)

        # Check total iteration
        self.iter = (start_epoch - 1) * self.num_batch
        self.pf = []
        self.partition = False
        self.total_loss = 0
        self.total_len = 0
        # self.total_metrics = {}
        self.dambda = 1
        self.step = 1
        self.temp = 512

        # Evaluation before training
        self.epoch = 0
        if getattr(args, "pre_evaluation", False):
            self.log.info(f"Epoch {self.epoch} / {self.train_arg.max_epoch}:")
            eval_loader_autodevice = DataPrefetcher(
                eval_loader, device=self.device
            )
            (
                dev_f1_metric,
                _,
                dev_ll,
                dev_left_metric,
                dev_right_metric,
            ) = self.evaluate(
                eval_loader_autodevice,
                decode_type=args.test.decode,
                eval_depth=eval_depth,
                left_binarization=self.left_binarization,
                right_binarization=self.right_binarization,
                rule_update=True,
            )
            self.log.info(f"{'dev f1:':6}   {dev_f1_metric}")
            self.log.info(f"{'dev ll:':6}   {dev_ll}")

            # Logging
            self.log_per_epoch(
                dev_f1_metric, dev_ll, dev_left_metric, dev_right_metric
            )

        # Watch model
        if getattr(args, "wandb", False):
            if getattr(args.wandb, "debug", False):
                self.watch_model = True
                wandb.watch(self.model, log="all", log_freq=1)
            else:
                self.watch_model = False

        for epoch in range(start_epoch, train_arg.max_epoch + 1):
            """
            Auto .to(self.device)
            """
            self.epoch = epoch
            self.temp = self.temp / 2

            # Warmup for epoch
            if hasattr(self.train_arg, "warmup_epoch"):
                if epoch > self.train_arg.warmup_epoch:
                    self.partition = True

            # curriculum learning. Used in compound PCFG.
            if train_arg.curriculum:
                self.max_len = min(
                    train_arg.start_len
                    + int((epoch - 1) / train_arg.increment),
                    train_arg.max_len,
                )
                self.min_len = train_arg.min_len
            else:
                self.max_len = train_arg.max_len
                self.min_len = train_arg.min_len

            train_loader = dataset.train_dataloader(
                max_len=self.max_len, min_len=self.min_len
            )
            # if epoch == 1:
            self.num_batch = len(train_loader)
            self.total_iter = self.num_batch * train_arg.max_epoch

            train_loader_autodevice = DataPrefetcher(
                train_loader, device=self.device
            )
            eval_loader_autodevice = DataPrefetcher(
                eval_loader, device=self.device
            )

            start = datetime.now()
            inst = f"Epoch {self.epoch} / {self.train_arg.max_epoch}:"
            if hasattr(self, "lr_scheduler"):
                inst = f"LR={self.lr_scheduler.get_last_lr()};" + inst
            self.log.info(inst)

            ##############
            #  Training  #
            ##############
            self.train(train_loader_autodevice)

            if hasattr(self, "lr_scheduler"):
                self.lr_scheduler.step()

            ##############
            # Evaluation #
            ##############
            (
                dev_f1_metric,
                _,
                dev_ll,
                dev_left_metric,
                dev_right_metric,
            ) = self.evaluate(
                eval_loader_autodevice,
                decode_type=args.test.decode,
                eval_depth=eval_depth,
                left_binarization=self.left_binarization,
                right_binarization=self.right_binarization,
                rule_update=True,
            )
            self.log.info(f"{'dev f1:':6}   {dev_f1_metric}")
            self.log.info(f"{'dev ll:':6}   {dev_ll}")

            # Logging
            self.log_per_epoch(
                dev_f1_metric, dev_ll, dev_left_metric, dev_right_metric
            )

            t = datetime.now() - start

            # save the model if it is the best so far
            if dev_ll > best_metric:
                best_metric = dev_ll
                best_e = epoch

                reproducibility.save(
                    args.save_dir + f"/best.pt",
                    model=self.model.state_dict(),
                    optimizer=self.optimizer.state_dict(),
                    epoch=epoch,
                )
                self.log.info(f"{t}s elapsed (saved)\n")
            else:
                self.log.info(f"{t}s elapsed\n")

            # save the last model
            reproducibility.save(
                args.save_dir + f"/last.pt",
                model=self.model.state_dict(),
                optimizer=self.optimizer.state_dict(),
                epoch=epoch,
            )

            total_time += t
            if train_arg.patience > 0 and epoch - best_e >= train_arg.patience:
                if hasattr(self.train_arg, "change") and self.train_arg.change:
                    self.train_arg.change = False
                    self.partition = not self.partition
                    best_metric = LikelihoodMetric()
                    best_metric.total_likelihood = -float("inf")
                    best_metric.total = 1
                else:
                    break

        def check_idx(tree_path, tree_tag):
            existed_path = sorted(Path(tree_path).glob(f"{tree_tag}*.pt"))
            if existed_path:
                existed_idx = [
                    int(p.stem.split(tree_tag)[-1]) for p in existed_path
                ]
                max_idx = max(existed_idx)
                return max_idx + 1
            else:
                return 0

        # Final evaluation for training set
        if hasattr(self.args, "tree"):
            train_loader_autodevice = DataPrefetcher(
                train_loader, device=self.device
            )
            _, _, _, _, _ = self.evaluate(
                train_loader_autodevice,
                decode_type=args.test.decode,
                eval_depth=eval_depth,
                left_binarization=self.left_binarization,
                right_binarization=self.right_binarization,
            )

            tree_path = getattr(self.args.tree, "save_dir", self.args.save_dir)
            tree_tag = getattr(self.args.tree, "tag", self.args.model.name)
            idx = check_idx(tree_path, tree_tag)
            torch.save(self.parse_trees, f"{tree_path}/{tree_tag}{idx}.pt")
        # else:
        #     tree_path = self.args.save_dir
        #     tree_tag = self.args.model.name

        if getattr(args, "tensorboard", False):
            self.writer.flush()
            self.writer.close()

        if getattr(args, "wandb", False):
            self.run.finish()

        self.log.info("End Training.")
        self.log.info(f"The model is saved in the directory: {args.save_dir}")
