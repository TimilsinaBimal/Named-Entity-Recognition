from tensorflow.keras import models, layers


def create_model(num_vocab, num_tags):
    model = models.Sequential()
    model.add(layers.Embedding(num_vocab + 1, 112))
    model.add(layers.Bidirectional(layers.LSTM(112, return_sequences=True)))
    model.add(layers.Dropout(0.2))
    model.add(layers.TimeDistributed(
        layers.Dense(num_tags, activation='relu')))

    return model
