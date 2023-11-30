import numpy as np

from services import clear_dist_folder, create_dist_folder_structure, visualize_losses, move_weights
from helpers import load_images, load_metadata, normalize_images, normalize_metadata, save_generated_images, save_generated_weights
from building import build_and_compile_gan

# prompt = input("Enter a description to generate the image: ")

def prepare():
    move_weights()
    clear_dist_folder()
    create_dist_folder_structure()

def train(gan, generator, discriminator, normalized_images, epoch, epochs, batch_size, latent_dim, save_interval, d_losses_real, d_losses_fake, g_losses):
    noise = np.random.normal(0, 1, (batch_size, latent_dim))

    generated_images = generator.predict(noise)

    idx = np.random.randint(0, normalized_images.shape[0], batch_size)

    real_images = normalized_images[idx]

    d_loss_real = discriminator.train_on_batch(real_images, np.ones((batch_size, 1)))
    d_loss_fake = discriminator.train_on_batch(generated_images, np.zeros((batch_size, 1)))
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    valid_labels = np.ones((batch_size, 1))
    g_loss = gan.train_on_batch(noise, valid_labels)

    print(f"Epoch {epoch}/{epochs} [D loss: {d_loss[0]} | D accuracy: {100 * d_loss[1]}] [G loss: {g_loss}]")

    if epoch % save_interval == 0:
        save_generated_images(epoch, generator, latent_dim)
        save_generated_weights(epoch, generator)

    d_losses_real.append(d_loss_real[0])
    d_losses_fake.append(d_loss_fake[0])
    g_losses.append(g_loss)


def main():
    prepare()

    train_images  = load_images('girls')
    # metadata = load_metadata('girls')

    normalized_images = normalize_images(train_images)
    # df_metadata, _ = normalize_metadata(metadata)

    d_losses_real = []
    d_losses_fake = []
    g_losses = []

    latent_dim = 100
    img_shape = (64, 64, 3)

    gan, generator, discriminator = build_and_compile_gan(latent_dim, img_shape)

    epochs = 10000
    batch_size = 64
    save_interval = 100

    for epoch in range(epochs):
        train(gan, generator, discriminator, normalized_images, epoch, epochs, batch_size, latent_dim, save_interval, d_losses_real, d_losses_fake, g_losses)
    
    visualize_losses(d_losses_real, d_losses_fake, g_losses)

main()