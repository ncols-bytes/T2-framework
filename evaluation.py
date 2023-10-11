import json
import time

from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import Sampler
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import mysql.connector
import numpy as np
import json
import re
import argparse
from multiprocessing import Pool
from utils.util import *
from data_process.data_process import *
from data_process.mysql_table_loader import *
from verification.verifiers import *
from model.configuration import TableConfig
from model.model import FilteringModel, VerificationModel


def load_ml_model(model_type, model_class, config_name, checkpoint_path, data_dir, device, ml_verif_tags):
    type_vocab = load_type_vocab(data_dir, model_type, ml_verif_tags)

    config_class = TableConfig
    config = config_class.from_pretrained(config_name)
    config.class_num = len(type_vocab)
    config.model_type = model_type

    model = model_class(config, is_simple=True)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    return model, type_vocab


def eval_single_dataset(dataset_name, args, device, data_processor, data_collator,
                        verif_conf,dict_verifiers,regex_verifiers,
                        model_f1,idx_2_tag_f1,type_vocab_f1,
                        model_f2,idx_2_tag_f2,type_vocab_f2):
    Y_true = []
    Y_pred_f1 = []
    Y_pred_f2 = []

    load_data_time_f1 = 0
    load_data_time_f2 = 0
    model_predict_time_f1 = 0
    model_predict_time_f2 = 0

    if dataset_name == 'wikitables':
        database = args.wikitables_database
    elif dataset_name == 'parent_tables':
        database = args.gittables1_database
    elif dataset_name == 'real_time_tables':
        database = args.gittables2_database

    mysql_table_loader = MysqlTableLoader(args.mysql_host, args.mysql_user, args.mysql_password, database)
    mysql_table_loader.connect()

    tables = mysql_table_loader.list_all_tables()

    entity_id_map = {}
    table_2_tags = {}
    
    table_need_f2_cnt = 0
    table_cnt = 0

    if dataset_name == 'wikitables':
        # load label and entity_id from original json
        with open(os.path.join(args.data_dir, "test.table_col_type.json"), "r") as fcc_file:
            fcc_data = json.load(fcc_file)
            for table_idx in range(len(fcc_data)):
                table_id = fcc_data[table_idx][0]
                entities = fcc_data[table_idx][6]
                annotations = fcc_data[table_idx][7]
                table_2_tags[table_id] = annotations

                entity_id_map[table_id] = {}
                for ci in range(len(entities)):
                    col_cell = entities[ci]
                    for cj in range(len(col_cell)):
                        index = col_cell[cj][0]
                        entity_id = col_cell[cj][1][0]
                        entity_id_map[table_id][str(index[0]) + '-' + str(index[1])] = entity_id

    # preload histograms
    load_histograms_start_time = time.time()
    histograms = mysql_table_loader.get_histograms()
    histogram_map = HistogramHelper().reformat_mysql_histograms(histograms)
    load_data_time_f1 += (time.time() - load_histograms_start_time)

    print(f'Evaluating {dataset_name}...')
    for ti in tqdm(range(len(tables)), desc="Processing"):
        table = tables[ti]
        table_cnt += 1

        # Phase1: get metadata
        load_data_start_time_f1 = time.time()

        table_name = table[0]
        table_id, pgTitle, pgEnt, secTitle, caption, headers, histogram = mysql_table_loader.get_metadata(table_name, histogram_map)

        load_data_time_f1 += (time.time() - load_data_start_time_f1)

        # Phase1: start filter
        model_predict_start_time_f1 = time.time()

        input_tok, input_tok_type, input_tok_pos, input_tok_tok_mask, input_tok_len, column_header_mask, \
                col_num, tokenized_meta_length, tokenized_headers_length, header_span, tokenized_pgTitle \
                = data_processor.process_single_table_metadata(pgTitle, secTitle, caption, headers)
        
        batch_size = 1
        max_input_tok_length = input_tok_len
        max_input_col_num = col_num

        input_tok, input_tok_type, input_tok_pos, input_tok_mask, column_header_mask, \
            histogram = data_collator.collate_metadata(batch_size, max_input_tok_length, max_input_col_num, [input_tok], [input_tok_type], \
                                    [input_tok_pos], [input_tok_tok_mask], [input_tok_len], \
                                    [column_header_mask], [col_num], [histogram])

        input_tok = input_tok.to(device)
        input_tok_type = input_tok_type.to(device)
        input_tok_pos = input_tok_pos.to(device)
        input_tok_mask = input_tok_mask.to(device)
        column_header_mask = column_header_mask.to(device)
        histogram = histogram.to(device)
        input_tok_mask = input_tok_mask[:,:,:input_tok_mask.shape[1]]

        with torch.no_grad():
            outputs = model_f1(input_tok, input_tok_type, input_tok_pos, input_tok_mask, column_header_mask, None, None, histogram)

        prediction_scores = outputs[0]
        prediction_labels = (torch.sigmoid(prediction_scores.view(-1, len(type_vocab_f1)))>0.5).tolist()
        prediction_labels = prediction_labels[:len(headers)]
        Y_pred_f1.extend(prediction_labels)

        model_predict_time_f1 += (time.time() - model_predict_start_time_f1)

        pred_tags_table_f2 = []
        if np.sum(prediction_labels) > 0:
            # start Phase2
            table_need_f2_cnt += 1
        
            # Phase2: get entity data
            load_data_start_time_f2 = time.time()
            entities = mysql_table_loader.get_entity_data(table_name, col_num, entity_id_map)
            load_data_time_f2 += (time.time() - load_data_start_time_f2)

            # Phase2: start all type verification
            model_predict_start_time_f2 = time.time()
            need_ml_verify_cols = []
            for col_idx in range(len(prediction_labels)):
                pred_tags_col_f2 = []

                pred_tags_f1 = [idx_2_tag_f1[i] for i, x in enumerate(prediction_labels[col_idx]) if x == True]
                for pred_tag in pred_tags_f1:
                    if verif_conf.get_verifier_type_by_tag(pred_tag) == 'dict':
                        ent_samples = [cell[1][1] for cell in entities[col_idx][:10]]
                        if dict_verifiers.verify(pred_tag, headers[col_idx], ent_samples) == True:
                            pred_tags_col_f2.append(pred_tag)
                    elif verif_conf.get_verifier_type_by_tag(pred_tag) == 'regex':
                        ent_samples = [cell[1][1] for cell in entities[col_idx][:10]]
                        if regex_verifiers.verify(pred_tag, ent_samples) == True:
                            pred_tags_col_f2.append(pred_tag)
                    elif verif_conf.get_verifier_type_by_tag(pred_tag) == 'ml':
                        need_ml_verify_cols.append(col_idx)
                pred_tags_table_f2.append(pred_tags_col_f2)

            # Phase2: start ml verification
            if len(need_ml_verify_cols) > 0:
                input_ent, input_ent_text, input_ent_cell_length, input_ent_type,  input_ent_mask, \
                    column_entity_mask, input_tok_ent_mask, input_ent_len \
                    = data_processor.process_single_table_entity_data(pgEnt, entities, col_num, input_tok_len, tokenized_meta_length,
                                        tokenized_headers_length, header_span, tokenized_pgTitle)
                
                max_input_ent_length = 10 * max_input_col_num + 1
                max_input_cell_length = 10

                input_ent_text, input_ent_text_length, input_ent, input_ent_type, input_ent_mask, \
                    column_entity_mask, input_tok_mask\
                    = data_collator.collate_entity_data(batch_size, max_input_tok_length, max_input_ent_length, max_input_cell_length, max_input_col_num, \
                                            [input_ent], [input_ent_text], [input_ent_cell_length], [input_ent_type], [input_ent_mask], [input_ent_len], \
                                            [column_entity_mask], [input_tok_tok_mask], [input_tok_ent_mask], [input_tok_len], [col_num])

                input_tok_mask = input_tok_mask.to(device)
                input_ent_text = input_ent_text.to(device)
                input_ent_text_length = input_ent_text_length.to(device)
                input_ent = input_ent.to(device)
                input_ent_type = input_ent_type.to(device)
                input_ent_mask = input_ent_mask.to(device)
                column_entity_mask = column_entity_mask.to(device)

                with torch.no_grad():
                    outputs = model_f2(input_tok, input_tok_type, input_tok_pos, input_tok_mask,\
                        input_ent_text, input_ent_text_length, input_ent, input_ent_type, input_ent_mask, column_entity_mask, column_header_mask, None, None, histogram)
                    prediction_scores = outputs[0]
                    prediction_labels = (torch.sigmoid(prediction_scores.view(-1, len(type_vocab_f2)))>0.5).tolist()
                    prediction_labels = prediction_labels[:len(headers)]

                    for col_idx in need_ml_verify_cols:
                        pred_tags_f2 = [idx_2_tag_f2[i] for i, x in enumerate(prediction_labels[col_idx]) if x == True]
                        pred_tags_table_f2[col_idx].extend(pred_tags_f2)

                model_predict_time_f2 += time.time() - model_predict_start_time_f2

        for col_idx in range(len(headers)):
            y_pred = [False] * len(type_vocab_f1)
            if len(pred_tags_table_f2) > 0:
                for label in pred_tags_table_f2[col_idx]:
                    y_pred[type_vocab_f1[label]] = True
            Y_pred_f2.append(y_pred)

        for col_idx in range(len(headers)):
            y_true = [False] * len(type_vocab_f1)
            if len(table_2_tags) > 0:
                for label in table_2_tags[table_id][col_idx]:
                    y_true[type_vocab_f1[label]] = True
            Y_true.append(y_true)

    mysql_table_loader.disconnect()

    Y_pred = np.array(Y_pred_f1)
    Y_true = np.array(Y_true)

    print(f"########## Evaluation of {dataset_name} ##########")
    if dataset_name == 'wikitables':
        print("========  Tagging Performance ========")
        print("--- Filtering ---")
        micro_precision = precision_score(Y_true, Y_pred, average='micro')
        micro_recall = recall_score(Y_true, Y_pred, average='micro')
        micro_f1 = f1_score(Y_true, Y_pred, average='micro')
        print("Micro-Precision:", micro_precision)
        print("Micro-Recall:", micro_recall)
        print("Micro-F1:", micro_f1)
        print("--- Overall ---")
        Y_pred = np.array(Y_pred_f2)
        micro_precision = precision_score(Y_true, Y_pred, average='micro')
        micro_recall = recall_score(Y_true, Y_pred, average='micro')
        micro_f1 = f1_score(Y_true, Y_pred, average='micro')
        print("Micro-Precision:", micro_precision)
        print("Micro-Recall:", micro_recall)
        print("Micro-F1:", micro_f1)
    else:
        print("======== Intrusiveness to Database ========")
        print("Ratio of scanned tables:", table_need_f2_cnt / table_cnt)

    print("======== Execution Time ========")
    print("--- Phase1 ---")
    print(f"Load metadata time: {load_data_time_f1}")
    print(f"Filtering model predict time: {model_predict_time_f1}")
    print("--- Phase2 ---")
    print(f"Load entity data time: {load_data_time_f2}")
    print(f"Verification model predict time: {model_predict_time_f2}")
    print()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mysql_host", default=None, type=str, required=True)
    parser.add_argument("--mysql_user", default=None, type=str, required=True)
    parser.add_argument("--mysql_password", default=None, type=str, required=True)
    parser.add_argument("--wikitables_database", default=None, type=str, required=True)
    parser.add_argument("--gittables1_database", default=None, type=str, required=True)
    parser.add_argument("--gittables2_database", default=None, type=str, required=True)
    parser.add_argument("--data_dir", default=None, type=str, required=True)
    parser.add_argument("--fitlter_model_conf", default=None, type=str, required=True)
    parser.add_argument("--fitlter_model_path", default=None, type=str, required=True)
    parser.add_argument("--verifi_model_conf", default=None, type=str, required=True)
    parser.add_argument("--verifi_model_path", default=None, type=str, required=True)
    parser.add_argument("--verif_conf", default=None, type=str, required=True)
    args = parser.parse_args()

    device = torch.device('cuda')

    entity_vocab = load_entity_vocab(args.data_dir, ignore_bad_title=True, min_ent_count=2)

    data_processor = DataProcessor(None, None, entity_vocab, type_vocab=None, histogram_len=1024, max_input_tok=500, src="test", max_length = [50, 10, 10], force_new=True, tokenizer = None)
    data_collator = DataCollator(None, data_processor.tokenizer, is_train=False)

    model_f1, type_vocab_f1 = load_ml_model(1, FilteringModel, args.fitlter_model_conf, args.fitlter_model_path, args.data_dir, device, None)
    print('load filtering model finished.')

    verif_conf = VerificationConfig(args.verif_conf)
    dict_verifiers = DictVerifiers(args.data_dir)
    regex_verifiers = RegexVerifiers()
    ml_verif_tags = verif_conf.get_tags_by_verifier_type('ml')  

    model_f2, type_vocab_f2 = load_ml_model(2, VerificationModel, args.verifi_model_conf, args.verifi_model_path, args.data_dir, device, ml_verif_tags)
    print('load verification model finished.')

    idx_2_tag_f1 = {}
    idx_2_tag_f2 = {}
    for tag, tag_idx in type_vocab_f1.items():
        idx_2_tag_f1[tag_idx] = tag
    for tag, tag_idx in type_vocab_f2.items():
        idx_2_tag_f2[tag_idx] = tag

    eval_single_dataset("wikitables", args, device, data_processor, data_collator, \
                        verif_conf,dict_verifiers,regex_verifiers, \
                        model_f1,idx_2_tag_f1,type_vocab_f1, \
                        model_f2,idx_2_tag_f2,type_vocab_f2)

    eval_single_dataset("parent_tables", args, device, data_processor, data_collator, \
                        verif_conf,dict_verifiers,regex_verifiers, \
                        model_f1,idx_2_tag_f1,type_vocab_f1, \
                        model_f2,idx_2_tag_f2,type_vocab_f2)

    eval_single_dataset("real_time_tables", args, device, data_processor, data_collator, \
                        verif_conf,dict_verifiers,regex_verifiers, \
                        model_f1,idx_2_tag_f1,type_vocab_f1, \
                        model_f2,idx_2_tag_f2,type_vocab_f2)

if __name__ == "__main__":
    main()