import os, json, pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from pathlib import Path
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_images(folder_path, img_size=(64, 64)):
    images = []

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', folder_path)
    for root, directories, files in os.walk(path):
        for file in files:
            if Path(os.path.join(path, file)).suffix.lower() == '.jpg':
                file_path = os.path.join(path, file)
                try:
                    img = load_img(file_path, target_size=img_size)
                    img_array = img_to_array(img)
                    images.append(img_array)
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")

    return np.array(images)

def normalize_images(images):
    images = images.astype('float32') / 255.0
    images = (images - 0.5) * 2.0
    return images

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def normalize_metadata(metadata):
    flat_metadata = [flatten_dict(entry) for entry in metadata]
    df = pd.DataFrame(flat_metadata)

    label_encoders = {}
    for column in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'encoders')
        encoder_path = os.path.join(path, f'{column}_encoder.pkl')
        with open(encoder_path, 'wb') as le_file:
            pickle.dump(le, le_file)

    return df, label_encoders

def load_metadata(folder_path):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', folder_path)
    meta_path = os.path.join(path, 'meta.json')

    with open(meta_path) as f: 
        metadata = json.load(f) 
        return metadata

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
    plt.savefig(os.path.join(path, 'dist', 'images', f"gan_generated_image_epoch_{epoch}.png"))
    plt.close()

def save_generated_weights(epoch, generator):
    path = os.path.dirname(os.path.abspath(__file__))
    generator.save_weights(os.path.join(path, 'dist', 'weights', f"generator_weights_epoch_{epoch}.h5"))

def create_metadata(folder_path):
    metadata = []

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', folder_path)
    meta_path = os.path.join(path, 'meta.json')

    if os.path.isfile(meta_path):
        with open(meta_path) as f: 
            metadata = json.load(f) 

    for root, _, files in os.walk(path):
        for file in files:
            if Path(os.path.join(path, file)).suffix.lower() == '.jpg':
                add_to_metadata = True
                for obj in metadata:
                    if obj['filename'] == file:
                        add_to_metadata = False
                        break

                if add_to_metadata:
                    metadata.append({"filename": file})

    with open(os.path.join(root, 'meta.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

def update_metadata(folder_path):
    metadata = []

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', folder_path)
    metaPath = os.path.join(path, 'meta.json')

    if os.path.isfile(metaPath):
        with open(metaPath) as f: 
            metadata = json.load(f) 

    for root, _, files in os.walk(path):
        for file in files:
            if Path(os.path.join(path, file)).suffix.lower() == '.jpg':
                add_to_metadata = True
                for obj in metadata:
                    obj['gender'] = obj.get('gender', '')
                    obj['hair'] = obj.get('hair', { "color": '', "hairstyle": ''})
                    obj['age'] = obj.get('age', '')
                    obj['location'] = obj.get('location', { "light": '', "place": '', "indoors": False })
                    obj['frame_filling'] = obj.get('frame_filling', 0)
                    obj['pose'] = obj.get('pose', { "faceTo": '', "bodyTo": '', "cameraFrom": '' })
                    obj['outfit'] = obj.get('outfit', { "top": '', "bottom": '', "head": '', "face": '' })

    with open(os.path.join(root, 'meta.json'), 'w') as f:
        json.dump(metadata, f, indent=2)