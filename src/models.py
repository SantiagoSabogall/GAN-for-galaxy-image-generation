import tensorflow as tf
from tensorflow.keras import layers
from .config import *

def build_generator():
    """
    Builds the Generator model suitable for 64x64x3 images.
    Uses ReLU activations and BatchNormalization.
    """
    model = tf.keras.Sequential([
        # Project and reshape (4x4x512)
        layers.Dense(4 * 4 * 512, use_bias=False, input_shape=(LATENT_DIM,)),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Reshape((4, 4, 512)),
        
        # 4x4 -> 8x8
        layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        
        # 8x8 -> 16x16
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        
        # 16x16 -> 32x32
        layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        
        # 32x32 -> 64x64
        layers.Conv2DTranspose(CHANNELS, (4, 4), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
    ], name="generator")
    return model

def build_discriminator():
    """
    Builds the Discriminator/Critic model for WGAN-GP.
    No BatchNormalization is used because it interferes with Gradient Penalty.
    Uses LeakyReLU for robust gradients.
    """
    model = tf.keras.Sequential([
        # 64x64 -> 32x32
        layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same', input_shape=[IMG_HEIGHT, IMG_WIDTH, CHANNELS]),
        layers.LeakyReLU(alpha=0.2),
        
        # 32x32 -> 16x16
        layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same'),
        layers.LeakyReLU(alpha=0.2),
        
        # 16x16 -> 8x8
        layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same'),
        layers.LeakyReLU(alpha=0.2),
        
        # 8x8 -> 4x4
        layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same'),
        layers.LeakyReLU(alpha=0.2),
        
        layers.Flatten(),
        # Output a single linear value for WGAN (no sigmoid)
        layers.Dense(1)
    ], name="discriminator")
    return model
