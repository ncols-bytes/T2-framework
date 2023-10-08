# T2-framework

This repository provides code for T2:Two-phase Tagging auto tagging framework.

## Environment

For convenience, you can use the following docker image to run this repository.

```
docker pull xdeng/transformers:latest
```

## Prepare Data

1. Download the following files with the same name from [this link](https://buckeyemailosu-my.sharepoint.com/personal/deng_595_buckeyemail_osu_edu/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fdeng%5F595%5Fbuckeyemail%5Fosu%5Fedu%2FDocuments%2FBuckeyeBox%20Data%2FTURL%2Fdata) and store them in the `data/wikitables_v2` directory.

```
├── data
    └── wikitables_v2
        ├── train.table_col_type.json
        ├── dev.table_col_type.json
        ├── test.table_col_type.json
        ├── entity_vocab.txt
    		└── type_vocab.txt
```

2. Download the pre-trained checkpoint from [this link](https://buckeyemailosu-my.sharepoint.com/personal/deng_595_buckeyemail_osu_edu/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fdeng%5F595%5Fbuckeyemail%5Fosu%5Fedu%2FDocuments%2FBuckeyeBox%20Data%2FTURL%2Fcheckpoint%2Fpretrained) and store it in the `checkpoints/pretrained_hybrib_model` directory.

```
├── checkpoints
    └── pretrained_hybrib_model
        └── pytorch_model.bin 
```

## Run T2

### Step1: Fine tune Filtering Model

```
./step1_finetune_filtering_model.sh
```

### Step2: Fine tune Verification Model

```
./step2_finetune_verification_model.sh
```

### Step3: Build MySQL table for testing

```
./step3_build_mysql_table_for_test.sh
```

### Step4: Evaluation

```
./step4_evaluation.sh
```

