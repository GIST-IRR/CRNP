## Post-processing

### String to Tree

Transform parse trees in string format to NLTK Trees and save to file.

```bash
python -m postprocessing.string_to_tree \
--filepath parsed/pftnpcfg_eng_nt30_t60_test.txt \
--vocab [path_to_log_dir]/word_vocab.pkl \
--output nltk_tree/pftnpcfg_eng_nt30_t60_test.pkl
```

### String to Span

Transform parse trees in string format to spans and save to file.

```bash
python -m postprocessing.string_to_span \
--filepath parsed/pftnpcfg_eng_nt30_t60_test.txt \
--vocab [path_to_log_dir]/word_vocab.pkl \
--output span_tree/pftnpcfg_eng_nt30_t60_test.pkl
```