import pickle
from pathlib import Path
import argparse
from collections import Counter, defaultdict
import itertools

import numpy as np
import torch
from nltk import Tree

from utils import load_trees, span_to_tree, tree_to_span, sort_span, clean_word
from torch_support.metric import preprocess_span
from torch_support.metric import sentence_level_f1 as f1
from torch_support.metric import iou as IoU


def set_leaves(tree, words):
    tp = tree.treepositions("leaves")
    for i, p in enumerate(tp):
        tree[p] = str(words[i])


def pretty_print(tree):
    word = tree["word"]

    def _pretty_print(tree, word):
        t = span_to_tree(tree)
        set_leaves(t, word)
        t.pretty_print()

    _pretty_print(tree["gold"], word)
    _pretty_print(tree["pred1"], word)
    _pretty_print(tree["pred2"], word)


def main(args, max_len=40, min_len=2):
    # Load vocab
    if args.vocab is not None:
        with args.vocab.open("rb") as f:
            vocab = pickle.load(f)
    else:
        vocab = None
        raise ValueError("Vocab is required.")

    # Load gold parse trees
    if args.gold is not None:
        args.eval_gold = True
        gold = load_trees(args.gold, min_len, max_len, True, vocab)

    # Load predicted parse trees
    trees = load_trees(args.tree, min_len, max_len, True, vocab)

    # Correspondence
    pt_corr = defaultdict(lambda: defaultdict(int))
    unk_pt_corr = defaultdict(lambda: defaultdict(int))
    nt_corr = defaultdict(lambda: defaultdict(int))

    for i, ts in enumerate(trees):
        word = ts["word"]
        sentence = ts["sentence"]

        # Check all trees are same
        assert ts["word"] == word

        # Preprocess gold trees
        if args.eval_gold:
            gs = gold[i]
            # Check predicted trees and gold tree are same
            assert gs["sentence"] == sentence

            gt = gs["span"]
            gt = [s[:2] for s in gt]
            gt = sort_span(gt)

        # Pre-terminal Correspondence
        unk_mask = [True if w == 1 else False for w in gs["word"]]
        g_pos = [p[1] for p in gs["tree"].pos()]
        p_pos = [p[1].replace("'", "") for p in ts["tree"].pos()]
        for m, gp, pp in zip(unk_mask, g_pos, p_pos):
            if m:
                unk_pt_corr[gp][pp] += 1
            else:
                pt_corr[gp][pp] += 1

        # Nonterminal Correspondence
        g_nt = [s[:2] for s in gs["span"] if not s[2] == "ROOT"]
        for s in ts["span"]:
            k = s[:2]
            t_s = s[2].replace("'", "")
            try:
                idx = g_nt.index(k)
                g_s = gs["span"][idx + 1][2]
                nt_corr[g_s][t_s] += 1
            except:
                continue

    pt_corr = {
        k: {
            j: e
            for j, e in sorted(c.items(), key=lambda x: x[1], reverse=True)
        }
        for k, c in sorted(pt_corr.items())
    }
    unk_pt_corr = {
        k: {
            j: e
            for j, e in sorted(c.items(), key=lambda x: x[1], reverse=True)
        }
        for k, c in sorted(unk_pt_corr.items())
    }
    f1s = [np.mean(f) for f in f1s]

    # Accuracy of Fixed spans: common uncommon에 대한 내용으로 수정 및 정리
    # ratio_fixed = (correct_fixed + wrong_fixed) / sum(num_fluct)
    # fixed_acc = correct_fixed / (correct_fixed + wrong_fixed)
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tree", required=True, type=Path)
    parser.add_argument("-g", "--gold", default=None, type=Path)
    parser.add_argument("-v", "--vocab", default=None, type=Path)
    args = parser.parse_args()

    main(args)
