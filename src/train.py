from sklearn.model_selection import train_test_split
from models.simple_keras_model import simpleModel, MODEL_NAME, MODEL_PATH
from utils import load_dataframe, transform_dataset, save_submission, save_history
from callbacks import build_save_callbacks
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf
from processing import isolate, process
from keras.preprocessing.image import ImageDataGenerator

BATCH_SIZE = 24
EPOCHS = 100
VERBOSE = 1

LEARNING_RATE = 0.001
BETA_1 = 0.9
BETA_2 = 0.999
EPSILON = 1e-08
DECAY = 0.0

def train_model(modelBuilder):
    train_df = load_dataframe('train')
    test_df = load_dataframe('test')

    X_train = process(transform_dataset(train_df), isolate)
    X_test = process(transform_dataset(test_df), isolate)

    target_train = train_df['is_iceberg']
    X_train_cv, X_valid, y_train_cv, y_valid = train_test_split(
        X_train,
        target_train,
        random_state=1,
        train_size=0.75
    )

    model = modelBuilder()
    optimizer = Adam(
        lr=LEARNING_RATE,
        beta_1=BETA_1,
        beta_2=BETA_2,
        epsilon=EPSILON,
        decay=DECAY
    )
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    model.summary()

    callbacks = build_save_callbacks(filepath=MODEL_PATH, patience=5)

    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )
    datagen.fit(X_train)

    empty = ImageDataGenerator()
    empty.fit(X_valid)

    steps_per_epoch = len(X_train_cv) // BATCH_SIZE
    hist = model.fit_generator(
        datagen.flow(X_train_cv, y_train_cv, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        verbose=VERBOSE,
        validation_data=empty.flow(X_valid, y_valid),
        steps_per_epoch=steps_per_epoch*2,
        callbacks=callbacks)

    model.load_weights(filepath=MODEL_PATH)
    score = model.evaluate(X_valid, y_valid, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    predicted_test = model.predict_proba(X_test)

    save_submission(test_df, predicted_test, filename='sub.csv')
    save_history(hist.history, model_name=MODEL_NAME)


if __name__ == '__main__':
    with tf.device('/gpu:1'):
        try:
            train_model(simpleModel)
        finally:
            K.clear_session()
