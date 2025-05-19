import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
absl.logging.set_stderrthreshold('error')

import numpy as np
import cv2
import argparse
from pathlib import Path
from glob import glob
from tqdm import tqdm
import logging

logger = logging.getLogger("pixal")

class ImageDataProcessor:
    def __init__(self, image_folders, pool_size=8, channels=("H", "S", "V"), file_name="out.npz", quiet=False):
        self.image_folders = image_folders
        self.pool_size = pool_size
        self.image_shape = None 
        self.channels = channels
        self.quiet = quiet
        self.file_name = file_name

    def find_divisible_size(self, h, w):
        new_h = h - (h % self.pool_size)
        new_w = w - (w % self.pool_size)
        return new_h, new_w

    def apply_average_pooling(self, v_channel):
        h, w = v_channel.shape
        new_h, new_w = self.find_divisible_size(h, w)
        self.image_shape = (new_h // self.pool_size, new_w // self.pool_size)
        v_channel = v_channel[:new_h, :new_w]
        pooled = v_channel.reshape(new_h // self.pool_size, self.pool_size, 
                                   new_w // self.pool_size, self.pool_size).mean(axis=(1, 3))
        return pooled

    def process_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            if not self.quiet:
                logger.warning(f"Error loading image: {image_path}")
            return None

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        channel_map = {
            "H": hsv_image[:, :, 0],
            "S": hsv_image[:, :, 1],
            "V": hsv_image[:, :, 2],
            "R": rgb_image[:, :, 0],
            "G": rgb_image[:, :, 1],
            "B": rgb_image[:, :, 2],
        }

        pooled_channels = []
        for ch in self.channels:
            if ch in channel_map:
                pooled = self.apply_average_pooling(channel_map[ch])
                pooled_channels.append(pooled.reshape(-1, 1))
            else:
                if not self.quiet:
                    logger.warning(f"Warning: Unknown channel '{ch}' requested.")

        if not pooled_channels:
            return None

        combined = np.concatenate(pooled_channels, axis=1)
        return combined

    def process_images_in_folder(self, folder_path):
        image_paths = glob(os.path.join(folder_path, "*"))
        all_images_data = []

        for image_path in tqdm(image_paths, desc=f"Processing {Path(folder_path).name}", disable=self.quiet):
            image_data = self.process_image(image_path)
            if image_data is not None:
                all_images_data.append(image_data)

        if not all_images_data:
            if not self.quiet:
                logger.warning(f"No valid images processed in {folder_path}.")
            return np.array([])

        all_images_data = np.array(all_images_data)
        if not self.quiet:
            logger.info(f"Processed {len(all_images_data)} images from {folder_path}, shape: {all_images_data.shape}")
        return all_images_data

    def load_and_label_data(self):
        data = []
        labels = []

        for idx, folder_path in enumerate(self.image_folders):
            images = self.process_images_in_folder(folder_path)
            if images.size == 0:
                continue
            num_images = images.shape[0]
            data.append(images)
            labels.extend([idx] * num_images)

        if not data:
            if not self.quiet:
                logger.warning("No images processed from any folder.")
            return None, None

        data = np.vstack(data)
        labels = np.array(labels)

        if not self.quiet:
            logger.info(f"Labels shape before one-hot encoding: {labels.shape}")

        num_classes = len(self.image_folders)
        labels = np.eye(num_classes)[labels]

        if not self.quiet:
            logger.info(f"Final data shape: {data.shape}")
            logger.info(f"Final labels shape: {labels.shape}")

        return data, labels

    def save_data(self, output_dir):
        data, labels = self.load_and_label_data()
        if data is not None and labels is not None:
            output_file = output_dir / self.file_name
            if not self.quiet:
                logger.info(f"Image shape after pooling: {self.image_shape}")
            np.savez(output_file, 
                     data=data, 
                     labels=labels,
                     shape=self.image_shape)
            if not self.quiet:
                logger.info(f"Data and labels saved to {output_file}")

def run(input_dir, output_dir=None, config=None, quiet=False):
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    image_folders = [f for f in input_path.iterdir() if f.is_dir()]
    if not quiet:
        logger.info(f"🔍 Processing images from {len(image_folders)} folders...")

    pool_size = config.preprocessor.pool_size if config and hasattr(config.preprocessor, 'pool_size') else 4
    channels = config.preprocessor.channels if config and hasattr(config.preprocessor, 'channels') else ("H", "S", "V", "R", "G", "B")
    file_name = config.preprocessor.file_name if config and hasattr(config.preprocessor, 'file_name') else "out.npz"
  
    processor = ImageDataProcessor(image_folders, pool_size=pool_size, channels=channels, file_name=file_name, quiet=quiet)
    processor.save_data(output_dir)
