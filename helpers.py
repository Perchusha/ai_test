import os
import numpy as np
import pandas as pd

from PIL import Image
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_images(folder_path, img_size=(64, 64)):
    images = []

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', folder_path)
    for root, directories, files in os.walk(path):
        for file in files:
            file_path = os.path.join(path, file)
            try:
                img = load_img(file_path, target_size=img_size)
                img_array = img_to_array(img)
                images.append(img_array)
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")

    return np.array(images)


def load_csv_data(data_name):
    path = os.path.dirname(os.path.abspath(__file__))
    interviews_df = pd.read_csv(os.path.join(path, 'data', data_name), sep='\t')
    print(interviews_df)


def normalize_images(images):
    images = images.astype('float32') / 255.0
    images = (images - 0.5) * 2.0
    return images


def save_generated_images(epoch, generator, latent_dim, examples=10, dim=(1, 10), figsize=(10, 1)):
    noise = np.random.normal(0, 1, (examples, latent_dim))
    generated_images = generator.predict(noise)

    generated_images = 0.5 * generated_images + 0.5

    fig, axs = plt.subplots(dim[0], dim[1], figsize=figsize)
    axs = axs.flatten()

    for i in range(examples):
        img = generated_images[i, :, :, :]
        axs[i].imshow(img)
        axs[i].axis('off')

    path = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(path, 'result', 'saved_images', f"gan_generated_image_epoch_{epoch}.png"))
    plt.close()

def save_generated_weights(epoch, generator):
    path = os.path.dirname(os.path.abspath(__file__))
    generator.save_weights(os.path.join(path, 'result', 'saved_models', f"generator_weights_epoch_{epoch}.h5"))
