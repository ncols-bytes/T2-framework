# T2-framework

This repository provides code for T2, a two-phase tagging framework for relational data. Tags are semantic types annotated to data and can drive a wide range of applications, such as data asset discovery and searching, sensitive data recognition and protection, data analysis, and machine learning pipelines, etc. T2 is particularly effective and efficient in tagging relational data when dealing with a large number of tables from diverse customers in the cloud.

## Environment

For convenience, you can use the following docker image to run this repository.

```sh
# Download the image
$ docker pull neoffaa/t2-framework:v1

# Run a container from the image
$ docker run -itd -v <source_path>:<target_path> --name <container_name> --gpus all neoffaa/t2-framework:v1
```

In addition, you need to set up a MySQL server (8.0.x is preferred) and ensure that the docker container can access it.

## Prepare Data

### 1. WikiTables data

Download the following files with the same name from the `data` directory of [this link](https://buckeyemailosu-my.sharepoint.com/:f:/g/personal/deng_595_buckeyemail_osu_edu/EjZWRtslWX9CubQ92jlmNTgB74hxxXszy9BUaXG5OL5F-g?e=HN2qtD) and put them in the `data/wikitables_v2` directory at the root of the project.

```
├── data
    └── wikitables_v2
        ├── train.table_col_type.json
        ├── dev.table_col_type.json
        ├── test.table_col_type.json
        ├── entity_vocab.txt
    	└── type_vocab.txt
```

### 2. GitTables data

Download `parent_tables_licensed.zip` and `real_time_tables_licensed.zip` from [this link](https://zenodo.org/record/6517052), put them in the `data/gittables` directory and unzip them.

Unzip command:
```sh
$ unzip -q parent_tables_licensed.zip -d parent_tables_licensed
$ unzip -q real_time_tables_licensed.zip -d real_time_tables_licensed
```

The directory structure after unzipping: 
```
├── data
    └── gittables
        ├── parent_tables_licensed
            ├── 000002_39.parquet
            ├── 0000092.parquet
            └── ...
        └── real_time_tables_licensed
            ├── $DIS_1.parquet
            ├── $ROKU_1.parquet
            └── ...
```

### 3. Pre-trained word vectors (GloVe)
Download `glove.6B.zip` from [this link](https://nlp.stanford.edu/data/glove.6B.zip), put them in the `data/glove` directory and unzip them.

Unzip command:
```sh
$ unzip glove.6B.zip
```

The directory structure after unzipping: 
```
├── data
    └── glove
        ├── glove.6B.50d.txt
        └── ...
```

### 4. Pre-trained hybrid model
Download the pre-trained checkpoint from the `checkpoint/pretrained` directory of [this link](https://buckeyemailosu-my.sharepoint.com/:f:/g/personal/deng_595_buckeyemail_osu_edu/EjZWRtslWX9CubQ92jlmNTgB74hxxXszy9BUaXG5OL5F-g?e=HN2qtD) and put it in the `checkpoints/pretrained_hybrid_model` directory at the root of the project.

```
├── checkpoints
    └── pretrained_hybrid_model
        └── pytorch_model.bin 
```

## Run T2-framework

### 1. Filtering Model
Run the following commands to fine tune the Filtering Model:
```sh
CUDA_VISIBLE_DEVICES="0"
python finetuning.py \
    --do_train \
    --model_type=1 \
    --data_dir="data/wikitables_v2" \
    --hybrid_model_path="checkpoints/pretrained_hybrid_model" \
    --hybrid_model_config="configs/hybrid_model_config.json" \
    --focal_loss_alpha=0.5 \
    --use_histogram_feature \
    --output_dir="checkpoints/fitltering_model/with_hist/alpha0.5" \
    --evaluate_during_training \
    --overwrite_output_dir
```

Parameter explanations:
- `--do_train`: Run training.
- `--model_type`: 1-Filtering Model; 2-Verification Model.
- `--data_dir`: The input data directory.
- `--hybrid_model_path`: The path of pre-trained hybrid model.
- `--hybrid_model_config`: The config of pre-trained hybrid model.
- `--focal_loss_alpha`: The alpha value of focal loss in filtering model.
- `--use_histogram_feature`: Whether to use histogram feature.
- `--output_dir`: The model output directory.
- `--evaluate_during_training`: Run evaluation during training at each logging step.
- `--overwrite_output_dir`: Overwrite the content of the output directory.

### 2. Verification Model

#### Deep Learning Verification Model
Run the following commands to fine tune the deep learning verification model:
```sh
CUDA_VISIBLE_DEVICES="0" 
python finetuning.py \
    --do_train \
    --model_type=2 \
    --data_dir="data/wikitables_v2" \
    --hybrid_model_path="checkpoints/pretrained_hybrid_model" \
    --hybrid_model_config="configs/hybrid_model_config.json" \
    --verif_conf="verification/verif_conf_dl_kb.json" \
    --use_histogram_feature \
    --output_dir="checkpoints/verification_model/with_hist" \
    --evaluate_during_training \
    --overwrite_output_dir
```

Parameter explanations:
- `--do_train`: Run training.
- `--model_type`: 1-Filtering Model; 2-Verification Model.
- `--data_dir`: The input data directory.
- `--hybrid_model_path`: The path of pre-trained hybrid model.
- `--hybrid_model_config`: The config of pre-trained hybrid model.
- `--verif_conf`: The verification config that includes mapping relationships between tags and verifiers.
- `--use_histogram_feature`: Whether to use histogram feature.
- `--output_dir`: The model output directory.
- `--evaluate_during_training`: Run evaluation during training at each logging step.
- `--overwrite_output_dir`: Overwrite the content of the output directory.

#### Random Forest Verification Models
Run the following command to train random forest verification models:
```sh
python train_rf.py \
    --data_dir="./data" \
    --output_dir="checkpoints/random_forest_verifiers"
```

Parameter explanations:
- `--data_dir`: The input data directory, containing wikitables and GloVe, as detailed in the "Prepare Data" section.
- `--output_dir`: The models output directory.


It takes a relatively long time to train all random forest verifiers, or you can also directly download the pre-trained random forest verifiers from [this link](https://drive.google.com/file/d/1ckg1BzNYzRaa90BN_ojJ09DzR527lec-/view?usp=drive_link).

#### Knowledge-Based Verification Models
The knowledge-based verification models are generated in the `verification/verifiers.py` and do not require training.

### 3. MySQL tables for testing
Run the following commands to build MySQL tables for testing:
```sh
python build_mysql_table.py \
    --mysql_host=<mysql_host> \
    --mysql_port=<mysql_port> \
    --mysql_user=<mysql_user> \
    --mysql_password=<mysql_password> \
    --wikitables_database=wikitables \
    --git_parent_database=parent_tables \
    --git_real_time_database=real_time_tables \
    --data_dir="./data"
```
Parameter explanations:
- `--mysql_host`: The hostname or IP address of the MySQL server.
- `--mysql_port`: The port number of the MySQL server.
- `--mysql_user`: The MySQL username for the connection.
- `--mysql_password`: The password associated with the username.
- `--wikitables_database`: An empty database used to store wikitables. If it doesn't exist, the program will automatically create it.
- `--git_parent_database`: An empty database used to store the parent_tables from GitTables. If it doesn't exist, the program will automatically create it.
- `--git_real_time_database`: An empty database used to store the real_time_tables from GitTables. If it doesn't exist, the program will automatically create it.
- `--data_dir`: The data directory, containing wikitables and gittables data, as detailed in the "Prepare Data" section.

### 4. Evaluation
Run the following commands to get evaluation results:
```sh
python evaluation.py \
    --mysql_host=<mysql_host> \
    --mysql_port=<mysql_port> \
    --mysql_user=<mysql_user> \
    --mysql_password=<mysql_password> \
    --wikitables_database=wikitables \
    --git_parent_database=parent_tables \
    --git_real_time_database=real_time_tables \
    --data_dir="./data" \
    --hybrid_model_config="configs/hybrid_model_config.json" \
    --fitlter_model_path="checkpoints/fitltering_model/with_hist/alpha0.5/pytorch_model.bin" \
    --verifi_model_path="checkpoints/verification_model/with_hist/pytorch_model.bin" \
    --rf_models_path="checkpoints/random_forest_verifiers" \
    --use_histogram_feature \
    --verif_conf="verification/verif_conf_dl_kb.json" \
    --eval_dataset=wikitables \
    --enable_phase1
```
Parameter explanations:
- `--mysql_host`: The hostname or IP address of the MySQL server.
- `--mysql_port`: The port number of the MySQL server.
- `--mysql_user`: The MySQL username for the connection.
- `--mysql_password`: The password associated with the username.
- `--wikitables_database`: The database that stores wikitables.
- `--git_parent_database`: The database that stores parent_tables from GitTables.
- `--git_real_time_database`: The database that stores real_time_tables from GitTables.
- `--data_dir`: The original data directory, containing wikitables and GloVe, as detailed in the "Prepare Data" section.
- `--hybrid_model_config`: The config of pre-trained hybrid model.
- `--fitlter_model_path`: The fitltering model directory.
- `--verifi_model_path`: The deep learning verification model directory.
- `--use_histogram_feature`: Whether to use histogram feature. It should be consistent with the setting of model training.
- `--verif_conf`: The verification config that includes mapping relationships between tags and verifiers, with optional values:
    - `verification/verif_conf_dl_kb.json`: use the combination of deep learning models and knowledge-based models (DL-KB).
    - `verification/verif_conf_rf.json`: only use random forest-based verifiers (RF).
- `--eval_dataset`: The (mixed) dataset used for evaluation, with optional values: `wikitables`, `mix_wr`, `mix_wp`, `mix_wpr` or `mix_pr`.
- `--enable_phase1`: Whether to enable filtering phase. If not enabled, it will become `EC`, which means exhaustive check of all verification models .



After running successfully, it will print the evaluation results for the selected dataset, including tagging performance (only for wikitables), execution time and ratio of scanned tables.
