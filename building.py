import os
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU


def build_generator(latent_dim):
    model = Sequential()
    model.add(Dense(128 * 16 * 16, input_dim=latent_dim))
    model.add(Reshape((16, 16, 128)))
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2DTranspose(3, (4, 4), activation='tanh', padding='same'))  # Финальный слой с 3 каналами (RGB)
    return model

def build_discriminator(img_shape):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=img_shape))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model

def build_gan(generator, discriminator):
    model = Sequential([generator, discriminator])
    return model

def build_and_compile_gan(latent_dim, img_shape, learning_rate=0.0002, beta_1=0.5):
    discriminator = build_discriminator(img_shape)
    discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1), metrics=['accuracy'])
    discriminator.trainable = False

    generator = build_generator(latent_dim)

    gan = build_gan(generator, discriminator)
    gan.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1))

    rootDir = os.path.dirname(os.path.abspath(__file__))
    workingWeightsDir = os.path.join(rootDir, 'weights', 'working_weights.h5')
    generator.load_weights(workingWeightsDir)
    print("Model was created successfully")

    return gan, generator, discriminator