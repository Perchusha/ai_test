import tensorflow as tf
import numpy as np

from services import clear_result_folder, create_result_folder_structure
from helpers import load_images, normalize_images, save_generated_images, save_generated_weights
from building import build_generator, build_discriminator, build_gan

clear_result_folder()
create_result_folder_structure()

# prompt = input("Enter a description to generate the image: ")

train_images = load_images('girls')

normalized_images = normalize_images(train_images)

latent_dim = 100
img_shape = (64, 64, 3)

discriminator = build_discriminator(img_shape)
discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5), metrics=['accuracy'])
discriminator.trainable = False

generator = build_generator(latent_dim)

gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5))

epochs = 10000
batch_size = 64
save_interval = 100

mse = tf.keras.losses.MeanSquaredError()

for epoch in range(epochs):
    noise = np.random.normal(0, 1, (batch_size, latent_dim))

    generated_images = generator.predict(noise)

    idx = np.random.randint(0, normalized_images.shape[0], batch_size)
    real_images = normalized_images[idx]

    mse_loss = mse(real_images, generated_images)

    d_loss_real = discriminator.train_on_batch(real_images, np.ones((batch_size, 1)))
    d_loss_fake = discriminator.train_on_batch(generated_images, np.zeros((batch_size, 1)))
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    valid_labels = np.ones((batch_size, 1))
    g_loss = gan.train_on_batch(noise, valid_labels)

    print(f"Epoch {epoch}/{epochs} [D loss: {d_loss[0]} | D accuracy: {100 * d_loss[1]}] [G loss: {g_loss}] [MSE loss: {mse_loss}]")

    if epoch % save_interval == 0:
        save_generated_images(epoch, generator, latent_dim)
        save_generated_weights(epoch, generator)