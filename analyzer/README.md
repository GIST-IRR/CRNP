## Analysis

### Correlation between F1 and NLL

Each CSV Files should have the following format:

```
f1 score, likelihood
f1 score, likelihood
...
```

If you want to get a CSV file from evaluation CSV file, use the command below.

```bash
awk -F "," '{ print $2, $4 }' log/pftnpcfg_eng_nt30_t60/pftnpcfg_eng_nt30_t60.csv > pftnpcfg_results.csv
```

`scatter_with_hist.py`: `Fig. 2(a)` Visualization for correlation between F1 and LL for single model with histogram.

```bash
python -m analyzer.correlation.scatter_with_hist \
--filepath pftnpcfg_results.csv
```

`scatter_comparison.py`: `Fig. 2(b)` Visualization for correlation between F1 and LL for various models.

```bash
python -m analyzer.correlation.scatter_comparision \
--filepath pftnpcfg_results.csv ftnpcfg_results.csv \
--label PFTNPCFG FTNPCFG
```

### Trees

> [!WARNING]
> Some analyzing tools below are not completely work.
> It will be corrected soon.

`compare_trees.py`: `Tab. 1` Calculate F1 score and IoU score for given parse trees.

`rule_frequency.py`: `Fig. 5` Visualize sorted distribution for frequencies that observed rules in parse trees.

`common_uncommon_hist.py`: `Fig. 9` Visualize the degree of rareness for rules and the accuracy according to the degree of rareness.

### The number of Unique rules

Visualize the number of unique rules for each sentence length.

#### For single model in figure. (`Fig. 3(a)`) 

```bash
python3 -m analyzer.unique_rules \
--input "[CSV file path]" \
--output "[Target output file]"
```

#### For different models in same figure. (`Fig. 3(b)`)

Use same command with `Fig. 3(a)`, but CSV file have to involve `group id` column to distinguish each group.

#### For each language in different sub-figures. (`Fig. 7`)

The column `group id` represent as subtitle of figure.
The following `tick_size`, `legend_size`, `label_size` is recommended for this figure.

```bash
python3 -m analyzer.unique_rules \
--input "[CSV file path]" \
--output "[Target output file path]" \
--split_by_group \
--n_col 5 \
--n_row 2 \
--tick_size 17 \
--legend_size 17 \
--label_size 30
```

### Performance

Visualize the performance according to the combination of multi-parsers.

#### For absolute performance (`Fig. 10`)

```bash
python3 -m analyzer.homo_hetero \
--input "[CSV file path]" \
--output "[output file path]" \
```

#### For difference between pre-trained parsers and trained models (`Fig. 6`)

```bash
python3 -m analyzer.homo_hetero \
--input "[CSV file path]" \
--output "[output file path]" \
--difference
```
