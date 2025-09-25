# Collapse-Relaxing Neural Parameterization

Source code of EMNLP2025 Main [Probability Distribution Collapse: A Critical Bottleneck to Compact Unsupervised Neural Grammar Induction](https://arxiv.org/abs/2407.16181).
Our code is based on the [TN-PCFG](https://github.com/sustcsonglin/TN-PCFG).

> [!NOTE]
> The whole process is the same as [Parse-Focusing](https://github.com/GIST-IRR/Parse-Focusing)
> However, we repeat the process here for convenience.

## Setup

### Prepare environment 

```bash
conda create -n swpf python=3.11
conda activate swpf
pip install -r requirements.txt
```

### Prepare dataset

We use three datasets PTB, CTB, SPMRL.

If you need to download the datasets, please refer to [TN-PCFG](https://github.com/sustcsonglin/TN-PCFG).

If you want to reproduce model yourself, download the datasets to local directory `data/raw` and follow the procedures below.

> [!IMPORTANT]
> To preprocess the datasets, the each dataset file has the formatted name such as `[prefix]-train.txt`.


```bash
python -m preprocessing.preprocessing \
--dir data/raw \
--save_dir data/english \
--prefix english
```

### (Optional) Generating dataset for baseline

Build new dataset that composed with generated parse trees. \[`right-branched` / `left-brancehd` / `random` / `right-binarized` / `left-binarized`\] parse trees are generated for each sentence in the given dataset.

```bash
python -m preprocessing.generate_focused_parse \
--factor [right-binarized/left-binarized/random/right-branched/left-branched] \
--vocab [path_to_vocab] \
--input [path_to_dataset] \
--output [path_to_save]
```

If you want to generate datasets for all languages, factors, and splits (train, valid, test):

```bash
./generate_focused_parse.sh
```

You can include or exclude options for languages, factors and splits in the script.

## Train

### (Optional) Prepare parse trees from pre-trained parsers

Our model uses three sets of parse trees parsed by three different parsers (Structformer, NBL-PCFG, FGG-TNPCFG).

If you want to train the `Parse-Focused TN-PCFG` using the pre-trained parsers we use, you can download them [here](https://1drv.ms/f/s!AkEpgY1bYqmLlIsO-H38Xf4IZzf7tg?e=G4iA4O).

To follow the instructions for training with our configuration, place the files in the `pretrained/` directory.

### (Optional) Prepare pre-trained model

> [!NOTE]
> You can evaluate pre-trained model without pre-trained parsers.

You can download our pre-trained model from [here](https://1drv.ms/f/s!AkEpgY1bYqmLlIsPmZW4SVHBskt3Fg?e=ssikxV).
You can evaluate the performance of the model without any training.

### Train Collapse-Relaxing Neural Parameterization Model

**Parse-focused TN-PCFG**

```bash
python train.py \
--conf config/pftnpcfg_eng_nt30_t60.yaml
```

**CRNP**

```bash
python train.py \
--conf config/crnp_eng_nt30_t60.yaml
```

After training, the path to the save directory is printed. It may be downloaded at `log/crnp_eng_nt30_t60/CRNP[datetime]`.

## Evaluation

You can use a model that you download from us or trained by yourself for `path_to_log_dir`. 

```bash
python evaluate.py \
--load_from_dir [path_to_log_dir]
```
The CSV file with the results is saved in parent directory of `path_to_log_dir`. For instance, `log/crnp_eng_nt30_t60/crnp_eng_nt30_t60.csv`.

This CSV file has the following format:

```csv
save dir, sentence-level F1, corpus-level F1, likelihood, perplexity
```

## Parsing

> [!CAUTION]
> If you use a large grammar model (almost larger than NT 90 / T 180), Viterbi parsing may not work due to out-of-memory issues.

> [!NOTE]
> MBR decoding does not predict constituent symbol labels.
> For detailed analysis accroding to symbol labels, it is better to use Viterbi than MBR.

```bash
python parse.py \
--load_from_dir [path_to_log_dir] \
--dataset data/raw/english-test.txt \
--decode_type viterbi \
--output parsed/crnp_eng_nt30_t60_test.txt \
```

## Post-processing

The post-processing bash script is handled in [Post-Processing](postprocessing/README.md).

## Analysis

The Analysis bash script is handled in [Analyzer](analyzer/README.md).

## Contact

If you have any questions, please remains in `issues` or contact to me (jinwookpark2296@gmail.com).