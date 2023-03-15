import tensorflow as tf

from keras import layers, losses
from keras.models import Model


class Denoise(Model):
  def __init__(self):
    super(Denoise, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Input(shape=(13, 20, 1)),
      layers.Conv2D(64, 3, activation='relu', padding = 'same'),
      layers.Conv2D(32, 3, activation='relu', padding = 'same'),
      ])

    self.decoder = tf.keras.Sequential([
      layers.Conv2DTranspose(32, kernel_size=3, activation='relu', padding='same'),
      layers.Conv2DTranspose(64, kernel_size=3, activation='relu', padding='same'),
      layers.Conv2D(1, kernel_size=3, activation='sigmoid', padding='same')
      ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded


def train_autoencoder(train_ds_noisy, train_ds, val_ds_noisy, val_ds):
  autoencoder = Denoise()

  autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

  autoencoder.fit(train_ds_noisy, train_ds,
                  epochs=10,
                  shuffle=False, #already shuffled....this allows for still using labels in CNN
                  batch_size=32, 
                  validation_data=(val_ds_noisy, val_ds))

  return autoencoder


def denoise_data(autoencoder, data):
  encoded_imgs = autoencoder.encoder(data).numpy()
  decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()
  return decoded_imgs








