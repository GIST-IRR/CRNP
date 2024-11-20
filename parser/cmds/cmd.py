# -*- coding: utf-8 -*-
import math
from collections import defaultdict
from pathlib import Path
from itertools import chain

import torch
import torch.nn as nn
from tqdm import tqdm
from parser.helper.metric import LikelihoodMetric, UF1, UAS

from utils import (
    depth_from_tree,
    sort_span,
    span_to_tree,
    save_rule_heatmap,
)


class CMD(object):
    def __call__(self, args):
        self.args = args

    def lambda_update(self, train_arg):
        if train_arg.get("dambda_warmup"):
            if self.iter < train_arg.warmup_start:
                dambda = 0
            elif (
                self.iter >= train_arg.warmup_start
                and self.iter < train_arg.warmup_end
            ):
                dambda = 1 / (
                    1
                    + math.exp(
                        (-self.iter + train_arg.warmup_iter)
                        / (self.num_batch / 8)
                    )
                )
            else:
                dambda = 1
        elif train_arg.get("dambda_step"):
            bound = train_arg.total_iter * train_arg.dambda_step
            if self.iter < bound:
                dambda = 0
            else:
                dambda = 1
        else:
            dambda = 1
            # self.dambda = 0
        return dambda

    def log_step(self, iter, start=0, step=500):
        """Log metrics for each logging step.
        Logging step check by **start** and **step**.

        Args:
            iter (int): current iteration.
            start (int, optional): start step of iteration. Defaults to 0.
            step (int, optional): each logging step. Defaults to 500.
        """
        if not (iter != start and iter % step == 0):
            return

        # Log training loss
        self.writer.add_scalar("train/loss", self.total_loss / step, iter)
        # Log lambda for warm up
        self.writer.add_scalar("train/lambda", self.dambda, iter)

        metrics = self.total_metrics
        for k, v in metrics.items():
            self.writer.add_scalar(f"train/{k}", metrics[k] / step, iter)

        # initialize metrics
        self.total_loss = 0
        self.total_len = 0
        for k in metrics.keys():
            metrics[k] = 0

        if hasattr(self.model, "pf"):
            self.writer.add_histogram(
                "train/partition_number",
                self.model.pf.detach().cpu(),
                iter,
            )
            self.pf = []

        return

    def train(self, loader):
        self.model.train()
        train_arg = self.args.train

        # Make directory for saving heatmaps
        heatmap_dir = Path(self.args.save_dir) / "heatmap"
        if not heatmap_dir.exists():
            heatmap_dir.mkdir(parents=True, exist_ok=True)

        t = tqdm(
            loader,
            total=int(len(loader)),
            position=0,
            leave=True,
            desc="Training",
        )
        for x, y in t:
            # Parameter update
            if not hasattr(train_arg, "warmup_epoch") and hasattr(
                train_arg, "warmup"
            ):
                if self.iter >= train_arg.warmup:
                    self.partition = True

            # Oracle
            if hasattr(train_arg, "gold"):
                gold_tree = y[f"gold_tree_{train_arg.gold}"]
            else:
                gold_tree = None

            # Gradient zero
            self.optimizer.zero_grad()

            if self.partition:
                self.dambda = self.lambda_update(train_arg)
                # Soft gradients
                if self.dambda > 0:
                    loss, z_l = self.model.loss(
                        x, partition=self.partition, soft=True
                    )
                    t_loss = (loss + z_l).mean()
                else:
                    loss = self.model.loss(x, temp=self.temp)
                    z_l = None
                    t_loss = loss.mean()

                t_loss.backward()

                loss = z_l.mean() if z_l is not None else loss.mean()

            else:
                loss = self.model.loss(
                    x,
                    partition=self.partition,
                    gold_tree=gold_tree,
                    pos=y["pos"],
                )
                loss = loss.mean()

                loss.backward()

            # Gradient clipping
            if train_arg.clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), train_arg.clip
                )

            # Gradient update
            self.optimizer.step()

            # writer
            self.total_loss += loss.item()
            self.total_len += x["seq_len"].max().double()

            for k in self.total_metrics.keys():
                if not k in self.total_metrics:
                    self.total_metrics[k] = 0
                self.total_metrics[k] += self.model.metrics[k].mean().item()

            if hasattr(self.model, "pf"):
                self.pf = (
                    self.pf + self.model.pf.detach().cpu().tolist()
                    if self.model.pf.numel() != 1
                    else [self.model.pf.detach().cpu().tolist()]
                )

            if getattr(train_arg, "heatmap", False):
                if (
                    self.iter % int(self.total_iter / self.train_arg.max_epoch)
                    == 0
                ):
                    batched = (
                        True if self.model.rules["rule"].dim() == 4 else False
                    )
                    save_rule_heatmap(
                        self.model.rules,
                        dirname=heatmap_dir,
                        filename=f"rule_dist_{self.iter}.png",
                        batched=batched,
                    )
            self.log_step(self.iter, start=0, step=1000)

            # Check total iteration
            self.iter += 1
        return

    @torch.no_grad()
    def evaluate(
        self,
        loader,
        eval_dep=False,
        decode_type="mbr",
        model=None,
        eval_depth=False,
        left_binarization=False,
        right_binarization=False,
        rule_update=False,
    ):
        if model == None:
            model = self.model
        model.eval()

        metric_f1 = UF1(n_nonterms=self.model.NT, n_terms=self.model.T)
        metric_f1_left = UF1(n_nonterms=self.model.NT, n_terms=self.model.T)
        metric_f1_right = UF1(n_nonterms=self.model.NT, n_terms=self.model.T)
        metric_uas = UAS()
        metric_ll = LikelihoodMetric()

        print("decoding mode:{}".format(decode_type))
        print("evaluate_dep:{}".format(eval_dep))
        t = tqdm(
            loader,
            total=int(len(loader)),
            position=0,
            leave=True,
            desc="Validation",
        )

        depth = (
            self.args.test.depth - 2 if hasattr(self.args.test, "depth") else 0
        )
        # label = getattr(self.args.test, "label", False)
        label = True

        self.pf_sum = torch.zeros(depth + 1)
        self.sequence_length = defaultdict(int)
        self.estimated_depth = defaultdict(int)
        self.estimated_depth_by_length = defaultdict(lambda: defaultdict(int))
        self.parse_trees = []
        self.parse_trees_type = []

        for x, y in t:
            result = model.evaluate(
                x,
                decode_type=decode_type,
                eval_dep=eval_dep,
                depth=depth,
                label=label,
                rule_update=rule_update,
            )

            # Save sequence lengths
            for l in x["seq_len"]:
                self.sequence_length[l.item()] += 1

            # Save predicted parse trees
            result["prediction"] = list(map(sort_span, result["prediction"]))
            self.parse_trees += [
                {
                    "sentence": y["sentence"][i],
                    "word": x["word"][i].tolist(),
                    "gold_tree": y["gold_tree"][i],
                    "pred_tree": result["prediction"][i],
                }
                for i in range(x["word"].shape[0])
            ]

            # Calculate depth of parse trees
            predicted_trees = [span_to_tree(r) for r in result["prediction"]]
            for tree in predicted_trees:
                if tree not in self.parse_trees_type:
                    self.parse_trees_type.append(tree)
            s_depth = [depth_from_tree(t) for t in predicted_trees]

            for d in s_depth:
                self.estimated_depth[d] += 1

            for i, l in enumerate(x["seq_len"]):
                l, d = l.item(), s_depth[i]
                self.estimated_depth_by_length[l][d] += 1

            nonterminal = len(result["prediction"][0][0]) >= 3
            metric_f1(
                result["prediction"],
                y["gold_tree"],
                y["depth"] if eval_depth else None,
                lens=True,
                nonterminal=nonterminal,
            )
            if left_binarization:
                metric_f1_left(
                    result["prediction"],
                    y["gold_tree_left"],
                    y["depth_left"],
                    lens=True,
                    nonterminal=nonterminal,
                )
            if right_binarization:
                metric_f1_right(
                    result["prediction"],
                    y["gold_tree_right"],
                    y["depth_right"],
                    lens=True,
                    nonterminal=nonterminal,
                )
            if "depth" in result:
                self.pf_sum = (
                    self.pf_sum
                    + torch.sum(result["depth"], dim=0).detach().cpu()
                )
            metric_ll(result["partition"], x["seq_len"])

            if eval_dep:
                metric_uas(result["prediction_arc"], y["head"])

        sorted_type = defaultdict(list)
        for tree in self.parse_trees_type:
            tree_length = len(tree.leaves())
            sorted_type[tree_length].append(tree)

        return (
            metric_f1,
            metric_uas,
            metric_ll,
            metric_f1_left,
            metric_f1_right,
        )
