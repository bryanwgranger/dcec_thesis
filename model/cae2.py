import tensorflow as tf

from keras import layers, losses
from keras.models import Model

def Autoencoder():

        model = tf.keras.Sequential([
            layers.InputLayer((128,128,3)),
            layers.Conv2D(32, (5, 5), activation='elu', padding='same', strides=2, name='conv1'),
            layers.Conv2D(64, (5, 5), activation='elu', padding='same', strides=2, name='conv2'),
            layers.Conv2D(128, (3, 3), activation='elu', padding='same', strides=2, name='conv3'),
            layers.Flatten(),
            layers.Dense(32, activation='elu', name='embedding'),
            # layers.InputLayer((32,)),
            layers.Dense(32768, activation='elu'),
            layers.Reshape((16,16,128)),
            layers.Conv2DTranspose(128, kernel_size=3, strides=2, activation='elu', padding='same', name='deconv1'),
            layers.Conv2DTranspose(64, kernel_size=5, strides=2, activation='elu', padding='same', name='deconv2'),
            layers.Conv2DTranspose(32, kernel_size=5, strides=2, activation='elu', padding='same', name='deconv3'),
            layers.Conv2D(3, kernel_size=(3, 3), activation='elu', padding='same'),])

        return model

    # def call(self, x):
    #     encoded = self.encoder(x)
    #     decoded = self.decoder(encoded)
    #     return decoded



