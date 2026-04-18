import tensorflow as tf
import kagglehub
import glob
import os
from .config import *

def download_dataset():
    """Downloads dataset from Kaggle if not already present."""
    # Check if dataset already exists in data/raw
    existing_images = glob.glob(os.path.join(DATASET_PATH, '**', '*.jpg'), recursive=True)
    if len(existing_images) > 0:
        print(f"Dataset already present in {DATASET_PATH}. Found {len(existing_images)} images.")
        return DATASET_PATH

    print("Downloading dataset using kagglehub...")
    path = kagglehub.dataset_download(KAGGLE_DATASET)
    print(f"Dataset downloaded to cache: {path}")
    
    return path

def load_and_preprocess(path):
    img_raw = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_raw, channels=CHANNELS)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = tf.cast(img, tf.float32)
    # Normalize to [-1, 1]
    img = (img - 127.5) / 127.5
    return img

def get_dataset():
    dataset_dir = download_dataset()
    image_paths = glob.glob(os.path.join(dataset_dir, '**', '*.jpg'), recursive=True)
    
    if not image_paths:
        raise ValueError("No images found in the dataset directory.")
        
    print(f"Dataset size: {len(image_paths)} images.")
    
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.shuffle(buffer_size=10000) # Shuffle buffer large enough
    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset
