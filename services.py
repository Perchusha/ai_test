import os, shutil, glob
import matplotlib.pyplot as plt

def clear_dist_folder():
    dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dist")

    for f in os.listdir(dir):
        path = os.path.join(dir, f)
        try:
            shutil.rmtree(path)
        except OSError:
            os.remove(path)
    print("Dist folder cleared successfully")

def create_dist_folder_structure():
    dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dist")
    os.mkdir(os.path.join(dir, 'images'))
    os.mkdir(os.path.join(dir, 'weights'))
    print("Dist folder structure created successfully")

def move_weights():
    rootDir = os.path.dirname(os.path.abspath(__file__))
    savedWeightsDir = os.path.join(rootDir, 'dist', 'weights')
    workingWeightsDir = os.path.join(rootDir, 'weights')

    if not os.path.exists(workingWeightsDir):
        os.makedirs(workingWeightsDir)

    files = glob.glob(os.path.join(savedWeightsDir, "generator_weights_epoch_*.h5"))

    if files:
        epochs = [int(file.split('_')[-1].split('.')[0]) for file in files]
        max_epoch = max(epochs)
        source_path = os.path.join(savedWeightsDir, f"generator_weights_epoch_{max_epoch}.h5")
        destination_path = os.path.join(workingWeightsDir, "working_weights.h5")
        shutil.copy2(source_path, destination_path)
        print(f"Latest weights (epoch {max_epoch}) copied successfully.")
    else:
        print("No weights files found in the specified directory.")

def visualize_losses(d_losses_real, d_losses_fake, g_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(d_losses_real, label='Discriminator Loss (Real)')
    plt.plot(d_losses_fake, label='Discriminator Loss (Fake)')
    plt.plot(g_losses, label='Generator Loss')
    plt.legend()
    plt.title('Discriminator and Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()