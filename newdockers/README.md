To test the **EMTransformer** docker, use the following commands:

* docker run -it --gpus all --entrypoint=/bin/bash emtransformer
* cd /workspace/DLMatchers/EMTransformer
* python src/run_all.py --model_type=roberta --model_name_or_path=roberta-base --data_processor=DeepMatcherProcessor --data_dir=abt_buy --train_batch_size=16 --eval_batch_size=16 --max_seq_length=180 --num_epochs=10 --seed=2

To test the **GNEM** docker, use the following commands:

* docker run -it --gpus all --entrypoint=/bin/bash gnem
* cd /workspace/GNEM
* python train.py --seed 28 --log_freq 5 --lr 0.0001 --embed_lr 0.00002 --epochs 10 --batch_size 2 --tableA_path data/abt_buy/tableA.csv --tableB_path data/abt_buy/tableB.csv --train_path data/abt_buy/train.csv --test_path data/abt_buy/test.csv --val_path data/abt_buy/valid.csv --gpu 0 --gcn_layer 1 --test_score_type mean min max

To test the **Magellan** docker, use the following commands:
1q
* git clone https://github.com/anhaidgroup/py_entitymatching.git
* open a python prompt and execute following commands:
* import py_entitymatching as em
* import pandas as pd
* A = em.read_csv_metadata('/workspace/py_entitymatching/notebooks/vldb_demo/dblp_demo.csv', key='id')
* B = em.read_csv_metadata('/workspace/py_entitymatching/notebooks/vldb_demo/acm_demo.csv', key='id')=
* ab = em.AttrEquivalenceBlocker()
* C1 = ab.block_tables(A, B, 'paper year', 'paper year', l_output_attrs=['title', 'authors', 'paper year'], r_output_attrs=['title', 'authors', 'paper year'])
