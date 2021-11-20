import pickle
from preprocessing import load_data, Preprocessing, split_data, padding, one_hot_encode
from models import create_model
from train import train_model
from constants import FILEPATH, BATCH_SIZE, EPOCHS, MODEL_PATH, MODEL_FOLDER


def main():
    data = load_data(filepath=FILEPATH)
    preprocess = Preprocessing(data)
    word_vocab = preprocess.encoding('Word')
    tag_vocab = preprocess.encoding('Tag')
    pickle.dump(word_vocab, open(MODEL_FOLDER + 'word_vocab.pkl', 'wb'))
    pickle.dump(tag_vocab, open(MODEL_FOLDER + 'tag_vocab.pkl', 'wb'))
    words, tags = preprocess.prepare_data()
    X = padding(words)
    max_len = X.shape[1]
    pickle.dump(max_len, open(MODEL_FOLDER + 'max_len.pkl', 'wb'))
    y = padding(tags)
    y = one_hot_encode(y, len(tag_vocab))
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = create_model(len(word_vocab), len(tag_vocab))
    hist = train_model(model, X_train, y_train, (X_test, y_test),
                       batch_size=BATCH_SIZE, epochs=EPOCHS)
    model.save(MODEL_PATH)


if __name__ == "__main__":
    main()
