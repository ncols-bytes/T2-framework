import re
import numpy as np
import string


class Word2vecUtil():
    def __init__(self, glove_path):
        self.word2vec = {}

        with open(glove_path, 'r', encoding='utf-8') as file:
            for line in file:
                word, vector = line.strip().split(' ', 1)
                vector = list(map(float, vector.split(' ')))
                self.word2vec[word] = vector

    def get_tokenized_str(self, text):
        tokens = re.sub(r'[^a-zA-Z0-9]', ' ', text).split(' ')
        tokens = [token.lower() for token in tokens if token not in string.punctuation]
        return " ".join(tokens)

    def text_2_weighted_vector(self, text):
        words = text.split()
        vec_list = [self.word2vec[word] for word in words if word in self.word2vec]
        if not vec_list:
            return np.zeros(len(list(self.word2vec.values())[0]))
        weights = np.ones(len(vec_list))
        weighted_vec = np.average(vec_list, weights=weights, axis=0)
        return weighted_vec
