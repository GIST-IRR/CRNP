# TN-PCFG

source code of  NAACL2021 "PCFGs Can Do Better: Inducing Probabilistic Context-Free Grammars with Many Symbols" and ACL2021 main conference: "Neural Bilexicalized PCFG Induction"

The repository also contain faster implementations of:

- Compound (Neural PCFG): https://www.aclweb.org/anthology/P19-1228/
- Neural Lexicalized PCFG: https://www.aclweb.org/anthology/2020.tacl-1.42/



## Setup

prepare environment 

```
conda create -n pcfg python=3.7
conda activate pcfg
while read requirement; do pip install $requirement; done < requirements.txt 
```

prepare dataset

You can download the dataset from:  https://mega.nz/folder/OU5yiTjC#oeMYj1gBhqm2lRAdAvbOvw

PTB:  ptb_cleaned.zip / CTB and SPRML: ctb_sprml_clean.zip

You can directly use the propocessed pickle file or create pickle file by your own

```
python  preprocessing.py  --train_file path/to/your/file --val_file path/to/your/file --test_file path/to/your/file  --cache path/
```

After this, your data folder should look like this:

```
config/
   ├── tnpcfg_r500_nt250_t500_curriculum0.yaml
   ├── ...
  
data/
   ├── ptb-train-lpcfg.pickle    
   ├── ptb-val-lpcfg.pickle
   ├── ptb-test-lpcfg.pickle
   ├── ...
   
log/
fastNLP/
parser/
train.py
evaluate.py
preprocessing.py
```



## Train

**TN-PCFG**

python train.py  --conf tnpcfg_r500_nt250_t500_curriculum0.yaml

**Compound PCFG**

python train.py --conf cpcfg_nt30_t60_curriculum1.yaml

....

## Evaluation

For example, the saved directory should look like this:

```
log/
   ├── NBLPCFG2021-01-26-07_47_29/
   	  ├── config.yaml
   	  ├── best.pt
   	  ├── ...
```

python evaluate.py --load_from_dir log/NBLPCFG2021-01-26-07_47_29  --decode_type mbr --eval_dep 1 

## Out-of-memory

If you encounter OOM, you should adjust the batch size in the yaml file. Normally, for GPUs with 12GB memory, batch size=4~8 is ok, while for evaluation of NBL-PCFGs, you should set a smaller batch size (1 or 2).  

## Contact

If you have any question, plz contact bestsonta@gmail.com. 

## Citation

If these codes help you, plz cite our paper:

```
@misc{yang2021neural,
      title={Neural Bi-Lexicalized PCFG Induction}, 
      author={Songlin Yang and Yanpeng Zhao and Kewei Tu},
      year={2021},
      eprint={2105.15021},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

@inproceedings{yang-etal-2021-pcfgs,
    title = "{PCFG}s Can Do Better: Inducing Probabilistic Context-Free Grammars with Many Symbols",
    author = "Yang, Songlin  and
      Zhao, Yanpeng  and
      Tu, Kewei",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.naacl-main.117",
    pages = "1487--1498",
    abstract = "Probabilistic context-free grammars (PCFGs) with neural parameterization have been shown to be effective in unsupervised phrase-structure grammar induction. However, due to the cubic computational complexity of PCFG representation and parsing, previous approaches cannot scale up to a relatively large number of (nonterminal and preterminal) symbols. In this work, we present a new parameterization form of PCFGs based on tensor decomposition, which has at most quadratic computational complexity in the symbol number and therefore allows us to use a much larger number of symbols. We further use neural parameterization for the new form to improve unsupervised parsing performance. We evaluate our model across ten languages and empirically demonstrate the effectiveness of using more symbols.",
}
```









