# TVQA

This is a modified fork of the [[TVQA original](https://github.com/jayleicn/TVQA)] repository for reproducibility purposes.

## Replication of TVQA S+Q 

To replicate he effectiveness results for TVQA S+Q, please, follow the [[running](https://github.com/jayleicn/TVQA))]  instructions of the original repository (this uses by default the Subtitle stream).

**The following experiments require that the Replication is executed first (at least the preprocessing part).**

## Reproduction of BM25+extra model

**Step 1**: Install the required Python packages: 

`conda env create -f environment.yml`

**Step 2**: Split original train dataset into new train, new validation (keep original validation dataset as "test", because test labels are not available)
        `split_data.py`

**Step 3**: Preprocessing corpus and queries

  `python ir_preprocessing.py`

**Step 4**: Building indexes

  `python ir_indexing.py`

**Step 5**: Run TF-IDF baseline

`python ir_baseline_tfidf.py --data_split all`

**Step 6**: Extract features 

`python ir_gen_features.py --data_split all`

**Step 7**: LambdaMART default

 `python3 ir_hpo.py --default_config`

**Step 8**: LambdaMART wint HPO: RS and BOHB

  `hpo=rs; python3 ir_hpo.py --hpo_method $hpo --min_budget 100 --max_budget 100 --n_iterations 200 --n_workers 1`
  `hpo=bohb; python3 ir_hpo.py --hpo_method $hpo --min_budget 30 --max_budget 100 --n_iterations 200 --n_workers 1`

**Step 9**: LambdaMART testing model:

  `python3 ir_hpo.py --test --leaf 5 --tree 450 --lr 0.44`
