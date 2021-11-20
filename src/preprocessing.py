import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def load_data(filepath):
    data = pd.read_csv(filepath, encoding="ISO-8859-1")
    data['Word'] = data['Word'].apply(lambda column: column.lower())
    return data


class Preprocessing:
    def __init__(self, data) -> None:
        self.data = data
        self.vocab = set()
        self.mapping = dict()
        self.demapping = dict()

    def generate_mappings(self, column_name):
        self.vocab = self.data[column_name].unique()
        self.mapping = {data_map: idx for idx,
                        data_map in enumerate(sorted(self.vocab))}
        self.demapping = {idx: data_map for idx,
                          data_map in enumerate(sorted(self.vocab))}

    def encoding(self, column_name):
        self.generate_mappings(column_name)
        self.data[column_name
                  + '_idx'] = self.data[column_name].map(self.mapping)
        return self.vocab

    def decoding(self, column_name):
        self.generate_mappings(column_name)
        self.data[column_name
                  + '_idx'] = self.data[column_name].map(self.demapping)
        return self.vocab

    def prepare_data(self):
        self.data['Sentence #'].fillna(
            method='ffill', axis=0, inplace=True)
        data_agg = self.data.groupby(['Sentence #'], as_index=False)[
            'Word', 'POS', 'Tag', 'Word_idx', 'Tag_idx'].agg(lambda x: list(x))
        words = data_agg['Word_idx'].tolist()
        tags = data_agg['Tag_idx'].tolist()
        return words, tags


def padding(pad_item):
    item = pad_sequences(pad_item)
    return item.astype('float32')


def one_hot_encode(sequences, num_tags):
    sequence_list = np.array(
        [to_categorical(sequence, num_classes=num_tags)
            for sequence in sequences], dtype='object')
    return sequence_list.astype('float32')


def split_data(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test
