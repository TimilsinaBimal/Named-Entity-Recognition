from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy


def train_model(model, X, y, validation_data: tuple, batch_size, epochs):
    model.compile(
        loss=CategoricalCrossentropy(),
        optimizer=Adam(),
        metrics=['accuracy']
    )
    history_ = model.fit(X, y, batch_size=batch_size, epochs=epochs,
                         validation_data=validation_data, verbose=1)
    return history_
