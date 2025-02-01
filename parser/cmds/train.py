# -*- coding: utf-8 -*-
from datetime import datetime, timedelta

from parser.cmds.cmd import CMD
from parser.helper.metric import LikelihoodMetric, Metric
from parser.helper.loader_wrapper import DataPrefetcher
import torch

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
    def setup_logger(self, args):
        # Setup logger
        console_level = args.get("console_level", "INFO")
        self.log = get_logger(args, console_level=console_level)
        self.log.info(f"Seed: {args.seed}")
        self.log.info("Create the model")
        self.log.info(f"{self.model}\n")
        self.log.info(self.optimizer)

        # Setup tensorboard writer
        self.writer = SummaryWriter(args.save_dir)

        # Setup WandB
        # start a new wandb run to track this script
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
        ## Visualization
        # heatmap_dir = Path(self.args.save_dir) / "heatmap"
        # for k in self.total_metrics.keys():
        #     mp.Process(
        #         target=tensor_to_heatmap,
        #         args=(self.model.metrics[k],),
        #         kwargs={
        #             "dirname": heatmap_dir,
        #             "filename": f"{k}_{self.iter}.png",
        #         },
        #     )

        # F1 score for each epoch
        tag = "valid"

        unary_jsd = metric.pairwise_js_div(self.model.rules["unary"])
        metric_list = {
            "valid/epoch": self.epoch,
            "valid/avg_likelihood": dev_ll.score,
            "valid/perplexity": dev_ll.perplexity,
            "valid/f1": dev_f1_metric.sentence_uf1,
            "valid/exact": dev_f1_metric.sentence_ex,
            "valid/unary_local_ppl": metric.local_ppl(
                self.model.rules["unary"]
            ),
            "valid/unary_global_ppl": metric.global_ppl(
                self.model.rules["unary"]
            ),
            "valid/unary_jsd_arithmetic": unary_jsd.mean(),
            "valid/unary_jsd_geometric": unary_jsd.log().mean().exp(),
        }

        try:
            binary_jsd = metric.pairwise_js_div(
                self.model.rules["rule"].flatten(1)
            )
            metric_list.update(
                {
                    "valid/binary_local_ppl": metric.local_ppl(
                        self.model.rules["rule"].flatten(1)
                    ),
                    "valid/binary_global_ppl": metric.global_ppl(
                        self.model.rules["rule"].flatten(1)
                    ),
                    "valid/binary_jsd_arithmetic": binary_jsd.mean(),
                    "valid/binary_jsd_geometric": binary_jsd.log()
                    .mean()
                    .exp(),
                }
            )
        except:
            pass
        try:
            head_jsd = metric.pairwise_js_div(self.model.rules["head"])
            left_jsd = metric.pairwise_js_div(self.model.rules["left"].T)
            right_jsd = metric.pairwise_js_div(self.model.rules["right"].T)
            metric_list.update(
                {
                    "valid/head_local_ppl": metric.local_ppl(
                        self.model.rules["head"]
                    ),
                    "valid/head_global_ppl": metric.global_ppl(
                        self.model.rules["head"]
                    ),
                    "valid/head_jsd_arithmetic": head_jsd.mean(),
                    "valid/head_jsd_geometric": head_jsd.log().mean().exp(),
                    "valid/left_local_ppl": metric.local_ppl(
                        self.model.rules["left"].T
                    ),
                    "valid/left_global_ppl": metric.global_ppl(
                        self.model.rules["left"].T
                    ),
                    "valid/left_jsd_arithmetic": left_jsd.mean(),
                    "valid/left_jsd_geometric": left_jsd.log().mean().exp(),
                    "valid/right_local_ppl": metric.local_ppl(
                        self.model.rules["right"].T
                    ),
                    "valid/right_global_ppl": metric.global_ppl(
                        self.model.rules["right"].T
                    ),
                    "valid/right_jsd_arithmetic": right_jsd.mean(),
                    "valid/right_jsd_geometric": right_jsd.log().mean().exp(),
                }
            )
        except:
            pass

        if self.left_binarization:
            metric_list.update(
                {
                    "valid/f1_left": dev_left_metric.sentence_uf1,
                    "valid/exact_left": dev_left_metric.sentence_ex,
                }
            )
        if self.right_binarization:
            metric_list.update(
                {
                    "valid/f1_right": dev_right_metric.sentence_uf1,
                    "valid/exact_right": dev_right_metric.sentence_ex,
                }
            )

        for k, v in metric_list.items():
            self.writer.add_scalar(f"{tag}/{k}", v, self.epoch)
        self.run.log(metric_list)

        metric_dict = {
            "f1_length": dev_f1_metric.sentence_uf1_l,
            "Ex_length": dev_f1_metric.sentence_ex_l,
            "f1_left_length": dev_left_metric.sentence_uf1_l,
            "Ex_left_length": dev_left_metric.sentence_ex_l,
            "f1_right_length": dev_right_metric.sentence_uf1_l,
            "Ex_right_length": dev_right_metric.sentence_ex_l,
            # "f1_depth": dev_f1_metric.sentence_uf1_d,
            # "Ex_depth": dev_f1_metric.sentence_ex_d,
            # "f1_left_depth": dev_left_metric.sentence_uf1_d,
            # "Ex_left_depth": dev_left_metric.sentence_ex_d,
            # "f1_right_depth": dev_right_metric.sentence_uf1_d,
            # "Ex_right_depth": dev_right_metric.sentence_ex_d,
        }

        for k, v in metric_dict.items():
            for i, val in v.items():
                self.writer.add_scalar(f"{tag}/{k}", val, i)

        # distribution of estimated span depth
        self.estimated_depth = dict(sorted(self.estimated_depth.items()))
        for k, v in self.estimated_depth.items():
            self.writer.add_scalar(
                "valid/estimated_depth", v / dev_f1_metric.n, k
            )

        # Model weight norm
        if getattr(self.args.train, "vector_histogram", False):
            log_weight_histogram(self.writer, self.model, self.epoch)
        # Rule probability distribution projection
        if getattr(self.args.train, "rule_embeddings", False):
            log_rule_prob(self.writer, self.model, self.epoch)

        # Log parse tree
        # tree_idx = torch.randint(0, len(self.parse_trees), 1).item()
        tree_idx = 0
        tree = self.parse_trees[tree_idx]

        from nltk.tree.prettyprinter import TreePrettyPrinter

        gold_tree = span_to_tree_with_sent(tree["gold_tree"], tree["sentence"])
        pred_tree = span_to_tree_with_sent(tree["pred_tree"], tree["sentence"])
        self.writer.add_text(
            "gold_tree", TreePrettyPrinter(gold_tree).text(), self.epoch
        )
        self.writer.add_text(
            "pred_tree", TreePrettyPrinter(pred_tree).text(), self.epoch
        )

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
        # Update vocab size
        # Add three different unknown tokens
        args.model.update({"V": len(dataset.word_vocab)})

        # Setup model
        set_model_dir("parser.model")
        self.model = get_model_args(
            args.model, self.device, checkpoint["model"]
        )

        # Setup optimizer
        self.optimizer = get_optimizer_args(
            args.optimizer, self.model.parameters(), checkpoint["optimizer"]
        )

        # Setup logger
        self.setup_logger(args)

        total_time = timedelta()
        best_e, best_metric = 1, Metric()

        # Setup Validation
        eval_max_len = getattr(args.test, "max_len", None)
        eval_loader = dataset.val_dataloader(max_len=eval_max_len)

        """
        Training
        """
        train_arg = getattr(args, "train")
        test_arg = getattr(args, "test")
        self.train_arg = train_arg
        self.test_arg = test_arg

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

        # Token setup
        self.pad_token = dataset.word_vocab.word2idx["<pad>"]
        self.unk_token = dataset.word_vocab.word2idx["<unk>"]
        self.mask_token = dataset.word_vocab.word2idx.get("<mask>", None)

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
        wandb.watch(self.model, log="all", log_freq=1)

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
            self.log.info(f"Epoch {self.epoch} / {self.train_arg.max_epoch}:")
            ##############
            #  Training  #
            ##############
            self.train(train_loader_autodevice)

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

        self.writer.flush()
        self.writer.close()
        self.run.finish()
        self.log.info("End Training.")
        self.log.info(f"The model is saved in the directory: {args.save_dir}")
