import numpy as np
import cv2
import os
import argparse
from glob import glob

class ImageDataProcessor:
    def __init__(self, image_folders, pool_size=8):
        """
        :param image_folders: List of folders containing images
        :param pool_size: Pooling kernel size (e.g., 4 reduces size by 4x)
        """
        self.image_folders = image_folders
        self.pool_size = pool_size  # Pooling kernel size
        self.image_shape = None 

    def find_divisible_size(self, h, w):
        """Finds the closest height and width divisible by pool_size."""
        new_h = h - (h % self.pool_size)  # Make height divisible by pool_size
        new_w = w - (w % self.pool_size)  # Make width divisible by pool_size
        return new_h, new_w

    def apply_average_pooling(self, v_channel):
        """Manually applies average pooling on the V channel."""
        h, w = v_channel.shape
        new_h, new_w = self.find_divisible_size(h, w)  # Ensure dimensions are divisible
        #self.image_shape = new_h,new_w
        self.image_shape = (new_h // self.pool_size, new_w // self.pool_size)
        # Crop the image to the closest divisible size
        v_channel = v_channel[:new_h, :new_w]

        # Reshape and compute the average per block
        pooled = v_channel.reshape(new_h // self.pool_size, self.pool_size, 
                                   new_w // self.pool_size, self.pool_size).mean(axis=(1, 3))

        return pooled

    def process_image(self, image_path, use_channels=("H", "S", "V", "R", "G", "B")):
        """
        Process a single image: Extract selected HSV channels and apply average pooling.
        
        Args:
            image_path: Path to the image file.
            use_channels: Tuple of channels to include. Options: "H", "S", "V"

        Returns:
            Pooled image data of shape (pixels, len(use_channels))
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading image: {image_path}")
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

        for ch in use_channels:
            if ch in channel_map:
                pooled = self.apply_average_pooling(channel_map[ch])
                pooled_channels.append(pooled.reshape(-1, 1))  # Flatten to (pixels, 1)
            else:
                print(f"Warning: Unknown channel '{ch}' requested.")

        if not pooled_channels:
            return None

        combined = np.concatenate(pooled_channels, axis=1)  # Shape: (pixels, n_channels)
        return combined


    def process_images_in_folder(self, folder_path):
        """Process all images in a folder and return their pooled Value (V) channels."""
        image_paths = glob(os.path.join(folder_path, "*"))
        all_images_data = []

        for image_path in image_paths:
            image_data = self.process_image(image_path)
            if image_data is not None:
                all_images_data.append(image_data)

        if not all_images_data:
            print(f"No valid images processed in {folder_path}.")
            return np.array([])  # Return empty array if no images found

        # Convert list to NumPy array (Shape: [num_images, new_num_pixels, 1])
        all_images_data = np.array(all_images_data)
        print(f"Processed {len(all_images_data)} images from {folder_path}, shape: {all_images_data.shape}")
        return all_images_data

    def load_and_label_data(self):
        """Load images from multiple folders, label them, and return (data, labels) separately."""
        data = []
        labels = []

        for idx, folder_path in enumerate(self.image_folders):
            images = self.process_images_in_folder(folder_path)  # Shape: (num_images, reduced_num_pixels, 1)
            if images.size == 0:
                continue  # Skip empty folders
            num_images = images.shape[0]

            # Append image data
            data.append(images)

            # Assign **one** label per image
            labels.extend([idx] * num_images)  # Assign integer label per image type

        if not data:
            print("No images processed from any folder.")
            return None, None

        # Stack data into shape (total_images, reduced_num_pixels, 1)
        data = np.vstack(data)
        labels = np.array(labels)  # Shape: (total_images,)

        print("Labels shape before one-hot encoding:", labels.shape)

        # Convert labels to one-hot encoding: (total_images, num_classes)
        num_classes = len(self.image_folders)
        labels = np.eye(num_classes)[labels]

        print("Final data shape:", data.shape)      # Expected: (total_images, reduced_num_pixels, 1)
        print("Final labels shape:", labels.shape)  # Expected: (total_images, num_classes)

        return data, labels

    def save_data(self, output_file):
        """Process, combine, and save data & labels separately."""
        data, labels = self.load_and_label_data()
        
        if data is not None and labels is not None:
            # Extract filename without extension and enforce `.npz`
            output_file = os.path.splitext(output_file)[0] + ".npz"
            print("Image shape after pooling: ",self.image_shape)
            np.savez(output_file, 
                 data=data, 
                 labels=labels,
                 shape=self.image_shape)
            print(f"Data and labels saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Process images and combine RGB/HSV values with labels for model input.")
    parser.add_argument("image_folder1", type=str, help="Path to the first image folder (e.g., './images1').")
    parser.add_argument("image_folder2", type=str, help="Path to the second image folder (e.g., './images2').")
    parser.add_argument("output_file", type=str, help="Path to save the output .npy file (e.g., 'output_data.npy').")
    args = parser.parse_args()

    image_folders = [args.image_folder1, args.image_folder2]
    processor = ImageDataProcessor(image_folders)
    processor.save_data(args.output_file)


if __name__ == "__main__":
    main()