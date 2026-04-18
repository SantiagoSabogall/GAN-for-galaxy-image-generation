import matplotlib.pyplot as plt
import tensorflow as tf
import os
import glob
import imageio
from .config import OUTPUT_DIR, LATENT_DIM

def save_plot_images(predictions, epoch):
    """
    Saves a 4x4 grid of generated images.
    predictions: Tensor of shape (16, 64, 64, 3) 
    """
    fig = plt.figure(figsize=(8, 8))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        # Rescale from [-1, 1] to [0, 1]
        img = (predictions[i, :, :, :] * 0.5) + 0.5
        plt.imshow(img.numpy())
        plt.axis('off')

    filepath = os.path.join(OUTPUT_DIR, f'image_at_epoch_{epoch:04d}.png')
    plt.savefig(filepath)
    plt.close(fig)

def generate_gif(gif_name='galaxy_gan_evolution.gif'):
    """
    Creates a GIF from all the saved progress images.
    """
    anim_file = os.path.join(OUTPUT_DIR, gif_name)
    
    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob(os.path.join(OUTPUT_DIR, 'image_at_epoch_*.png'))
        filenames = sorted(filenames)
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
    print(f"Saved GIF to {anim_file}")
