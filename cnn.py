import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

from keras import layers, models, losses
from keras.callbacks import ModelCheckpoint
from keras.models import Model

from process_data import *              
from autoencoder import *         # current around 100%!!! with 20 folders of data and 10 epochs

# # import warnings for reading in data
# from warnings import simplefilter
# # ignore all future warnings
# simplefilter(action='ignore', category=FutureWarning)

# digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# train_ds, train_labels, train_ds_noisy, train_labels_noisy, test_ds, test_labels, test_ds_noisy, \
#         test_labels_noisy, val_ds, val_labels, val_ds_noisy, val_labels_noisy = preprocess_data(1)

# mfcc_shape = train_ds[0].shape

# autoencoder = train_autoencoder(train_ds_noisy, train_ds, val_ds_noisy, val_ds)
# train_decoded = denoise_data(autoencoder, train_ds_noisy)
# test_decoded = denoise_data(autoencoder, test_ds_noisy)
# val_decoded = denoise_data(autoencoder, val_ds_noisy)


# build the cnn
def create_cnn(shape):
        ip = layers.Input(shape=shape)

        m = layers.Conv2D(64, 3, activation='relu')(ip)
        m = layers.Conv2D(64, 3, activation='relu')(m)
        m = layers.MaxPooling2D()(m)
        m = layers.Dropout(0.2)(m)
        m = layers.Flatten()(m)
        m = layers.Dense(64, activation='relu')(m)
        m = layers.Dense(32, activation='relu')(m)
        m = layers.Dropout(0.5)(m)
        op = layers.Dense(10, activation='sigmoid')(m)

        model = tf.keras.Model(inputs=ip, outputs=op)
        return model


# train the cnn on the training data, returning the trained model
def train_model(model, train, train_labels, val, val_labels):
        # load weights
        model.load_weights("best_weights.hdf5")

        """
        data already checkpointed pretty accurately...uncomment to change during next run
        """

        # checkpoint_path="best_weights.hdf5"

        # # Create a callback that saves the model's weights
        # checkpoint = tf.keras.callbacks.ModelCheckpoint( \
        #     filepath=checkpoint_path,save_best_only=True,mode='max',monitor='val_accuracy',verbose=1)
        # callbacks_list = [checkpoint]

        model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

        history = model.fit(train,
                train_labels,
                epochs=10,
                batch_size=32,
                validation_data=(val, val_labels),
                # callbacks=[checkpoint], # uncomment when checkpointing again
                )

        # make a plot of accuracy
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

        # return the trained model
        return model


# use the trained model to classify the test data
def test_model(model, test, test_labels):
        y_pred= model.predict(test)
        y_p= np.argmax(y_pred, axis=1)
        y_pred=y_pred.astype(int)
        y_t=np.argmax(test_labels, axis=1)
        test_acc = sum(y_p == y_t) / len(y_t)

        print(f'Test set accuracy: {test_acc:.0%}')

        # plot accuracy in a cool figure
        confusion_mtx = tf.math.confusion_matrix(y_t, y_p) 
        plt.figure(figsize=(5, 5))
        sns.heatmap(confusion_mtx, 
                annot=True, fmt='g')
        plt.xlabel('Prediction')
        plt.ylabel('Label')
        plt.show()




