import os

# Dataset
DATASET_PATH = 'data/raw'
KAGGLE_DATASET = "jaimetrickz/galaxy-zoo-2-images"

# Image dimensions
IMG_HEIGHT = 64
IMG_WIDTH = 64
CHANNELS = 3

# Training Hyperparameters
BATCH_SIZE = 64
LATENT_DIM = 128
EPOCHS = 10
LEARNING_RATE_G = 0.0002
LEARNING_RATE_D = 0.0002
BETA_1 = 0.5
BETA_2 = 0.9

# WGAN-GP specific
N_CRITIC = 5 # Number of critic updates per generator update
GP_WEIGHT = 10.0 # Gradient penalty weight

# Paths
CHECKPOINT_DIR = './checkpoints'
LOG_DIR = './logs'
OUTPUT_DIR = './output'

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
# We don't make DATASET_PATH since it will be a symlink or handled by cache
