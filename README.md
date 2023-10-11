# T2-framework

This repository provides code for T2:Two-phase Tagging auto tagging framework.

## Environment

For convenience, you can use the following docker image to run this repository.

```sh
# Download the image
$ docker pull neoffaa/transformers:v1

# Run a container from the image
$ docker run -itd -v $(source_path):$(target_path) --name $(container_name) --gpus all neoffaa/transformers:v1
```

In addition, you need to set up a MySQL environment (8.0.x is preferred) and ensure that the docker container can access it.

## Prepare Data

### 1. WikiTables data

Download the following files with the same name from [this link](https://buckeyemailosu-my.sharepoint.com/personal/deng_595_buckeyemail_osu_edu/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fdeng%5F595%5Fbuckeyemail%5Fosu%5Fedu%2FDocuments%2FBuckeyeBox%20Data%2FTURL%2Fdata) and put them in the `data/wikitables_v2` directory.

```
├── data
    └── wikitables_v2
        ├── train.table_col_type.json
        ├── dev.table_col_type.json
        ├── test.table_col_type.json
        ├── entity_vocab.txt
    	└── type_vocab.txt
```

Please note that the `train.table_col_type.json` file in the link has been truncated, causing the JSON format to be corrupted. You need to modify the end of `train.table_col_type.json` to restore its correct format.

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
Download the pre-trained checkpoint from [this link](https://buckeyemailosu-my.sharepoint.com/personal/deng_595_buckeyemail_osu_edu/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fdeng%5F595%5Fbuckeyemail%5Fosu%5Fedu%2FDocuments%2FBuckeyeBox%20Data%2FTURL%2Fcheckpoint%2Fpretrained) and put it in the `checkpoints/pretrained_hybrid_model` directory.

```
├── checkpoints
    └── pretrained_hybrid_model
        └── pytorch_model.bin 
```

## Run T2-framework

### Step1: Fine tune Filtering Model
Run `step1_finetune_filtering_model.sh`:
```sh
$ ./step1_finetune_filtering_model.sh
```

After running successfully, it will generate a checkpoint file named `pytorch_model.bin` in the `checkpoints/fitltering_model` directory.


### Step2: Fine tune Verification Model
Run `step2_finetune_verification_model.sh`:
```sh
$ ./step2_finetune_verification_model.sh
```
After running successfully, it will generate a checkpoint file named `pytorch_model.bin` in the `checkpoints/verification_model` directory.

### Step3: Build MySQL table for testing
Replace the parameters in `step3_build_mysql_table_for_test.sh` and run it:
```sh
$ ./step3_build_mysql_table_for_test.sh
```
After running successfully, it will create 3 databases in MySQL.

### Step4: Evaluation
Replace the parameters in `step4_evaluation.sh` and run it:
```sh
$ ./step4_evaluation.sh
```
After running successfully, it will print the evaluation results.
