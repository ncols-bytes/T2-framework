import os
import pickle
import argparse
from utils.vocab_util import *
from data_process.data_processor import *
from data_process.mysql_table_loader import *
from verification.verification_config import *
from verification.domain_verifiers import *
from verification.rf_verifiers import *
from model.configuration import TableConfig
from model.model import FilteringModel, VerificationModel
from utils.word2vec_util import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None, type=str, required=True)
    parser.add_argument('--output_dir', default=None, type=str, required=True)
    args = parser.parse_args()

    wikitables_data_dir = os.path.join(args.data_dir, "wikitables_v2")
    type_vocab = load_type_vocab(wikitables_data_dir, 1, None)

    glove_path = os.path.join(args.data_dir, "glove/glove.6B.50d.txt")
    word2vec = Word2vecUtil(glove_path)
    rf_verifiers = RFVerifiers(word2vec)

    train_dataset = os.path.join(args.data_dir, "wikitables_v2/train.table_col_type.json")
    rf_verifiers.train(train_dataset, type_vocab, args.output_dir)
    
if __name__ == "__main__":
    main()