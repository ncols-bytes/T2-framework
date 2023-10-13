import os
import pickle
from utils.word2vec_util import *

import time
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
import json

from tqdm import tqdm

class RFVerifiers():
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.tag_2_model = {}

    def load_saved_models(self, saved_models_path):
        for root, dirs, files in os.walk(saved_models_path):
            for file_name in files:
                if '.pkl' not in file_name:
                    continue
                tag = file_name.replace('.pkl', '')
                
                file_path = os.path.join(root, file_name)
                with open(file_path, 'rb') as file:
                    classifier = pickle.load(file)
                    self.tag_2_model[tag] = classifier

    def extract_features(self, page_title, section_title, caption, header, ent_samples):
        title_tokens = self.word2vec.get_tokenized_str(page_title + ' ' + section_title + ' ' + caption)
        title_vec = self.word2vec.text_2_weighted_vector(title_tokens)

        distinct_set = set()
        max_len = 0
        avg_len = 0
        ent_tokens = ""
        
        for ent in ent_samples:
            distinct_set.add(ent)
            value_len = len(ent)
            max_len = max(max_len, value_len)
            avg_len += value_len
            ent_tokens += self.word2vec.get_tokenized_str(ent) + " "

        avg_len /= len(ent_samples)
        
        header_tokens = self.word2vec.get_tokenized_str(header)
        x = []
        x.extend(title_vec)
        x.extend(self.word2vec.text_2_weighted_vector(header_tokens)) 
        x.extend(self.word2vec.text_2_weighted_vector(ent_tokens)) 
        x.append(max_len)
        x.append(avg_len)
        x.append(len(distinct_set))

        return x

    def train(self, train_dataset, tag_2_idx, saved_model_path):
        X_train = []
        Y_train = []
        with open(train_dataset, 'r') as fcc_file:
            fcc_data = json.load(fcc_file)
            
            print(f'Extracting features...')
            for ti in tqdm(range(len(fcc_data)), desc="Processing"):
                                
                table_id = fcc_data[ti][0]
                page_title = fcc_data[ti][1]
                page_id = fcc_data[ti][2]
                section_title = fcc_data[ti][3]
                caption = fcc_data[ti][4]
                headers = fcc_data[ti][5]
                cells = fcc_data[ti][6]
                annotations = fcc_data[ti][7]

                for ci in range(len(headers)):
                    ent_samples = [cell[1][1] for cell in cells[ci][:10]]
                    x = self.extract_features(page_title, section_title, caption, headers[ci], ent_samples)
                    X_train.append(x)

                    y = [0] * len(tag_2_idx)
                    for tag in annotations[ci]:
                        y[tag_2_idx[tag]] = 1
                    Y_train.append(y)

        X_train_np = np.array(X_train)
        Y_train_np = np.array(Y_train)
        Y_train_np = np.transpose(Y_train_np)

        tag_cnt = 0
        if not os.path.exists(saved_model_path):
            os.makedirs(saved_model_path)
            
        for tag, tag_idx in tag_2_idx.items():
            start_time = time.time()
            print(f"start training tag:{tag}")
            
            over_sampler = RandomOverSampler(random_state=42)
            X_train_resampled, y_train_resampled = over_sampler.fit_resample(X_train_np, Y_train_np[tag_idx])
            # print(X_train_resampled.shape, y_train_resampled.shape)
            
            X_train, y_train = shuffle(X_train_resampled, y_train_resampled)
            classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            classifier.fit(X_train, y_train)
            
            with open(saved_model_path + '/' + tag + '.pkl', 'wb') as file:
                pickle.dump(classifier, file)
                
            tag_cnt += 1
            elapsed_time = int(time.time() - start_time)
            print(f"{tag_cnt}: {tag}, train finished, elapsed_time={elapsed_time}s")

    def verify(self, tag, page_title, section_title, caption, headers, cells):
        if tag not in self.tag_2_model:
            return None
        
        x = self.extract_features(page_title, section_title, caption, headers, cells)
        classifier = self.tag_2_model[tag]
        y_pred = classifier.predict([x])
        return y_pred[0] > 0.5
