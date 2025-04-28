import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os

def plot_mse_heatmap(model, X_test, y_test, output_dir="mse_plots"):
    """
    Computes per-pixel MSE and overlays an anomaly heatmap on the original images.
    
    Parameters:
        model: Trained Autoencoder model.
        X_test: Test images (flattened).
        y_test: Corresponding labels.
        output_dir: Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    predictions = model.predict([X_test, y_test])  # Predict reconstructed images

    # Compute per-pixel MSE
    mse = np.mean((X_test - predictions) ** 2, axis=1)  # Mean MSE per image
    pixel_mse = np.mean((X_test - predictions) ** 2, axis=0)  # Mean MSE per pixel across dataset

    print(f"Mean MSE across test set: {np.mean(mse):.6f}")

    # Iterate over images
    for i in range(min(5, len(X_test))):  # Show first 5 images
        original = X_test[i].reshape(90, 90)  # Reshape to image size (adjust as needed)
        reconstructed = predictions[i].reshape(90, 90)

        # Compute per-pixel error (reshaped)
        error_map = ((X_test[i] - predictions[i]) ** 2).reshape(90, 90)

        # Normalize error map for visualization
        norm_error_map = cv2.normalize(error_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap = cv2.applyColorMap(norm_error_map, cv2.COLORMAP_JET)

        # Overlay heatmap on the original image
        overlay = cv2.addWeighted(cv2.cvtColor(original.astype(np.uint8), cv2.COLOR_GRAY2BGR), 0.5, heatmap, 0.5, 0)

        # Save and display results
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(original, cmap="gray")
        axes[0].set_title("Original Image")
        axes[1].imshow(overlay)
        axes[1].set_title("Anomaly Heatmap")
        axes[2].imshow(reconstructed, cmap="gray")
        axes[2].set_title("Reconstructed Image")

        for ax in axes:
            ax.axis("off")

        plt.savefig(os.path.join(output_dir, f"mse_heatmap_{i}.png"))
        plt.show()

    return mse


def plot_mse_heatmap_overlay(model, X_test, y_test, image_shape, output_dir="analysis_plots", threshold=0.01):
    """
    Computes per-pixel MSE and overlays an anomaly heatmap on the original images.

    Parameters:
        model: Trained Autoencoder model.
        X_test: Test images (flattened).
        y_test: Corresponding one-hot labels.
        image_shape: Tuple representing the (height, width) of the original image.
        output_dir: Directory to save plots.
        threshold: MSE value above which pixels are considered anomalous.
    """
    os.makedirs(output_dir, exist_ok=True)

    predictions = model.predict([X_test, y_test])  # Predict reconstructed images

    for i in range(min(5, len(X_test))):  # Limit to first 5 examples
        original_flat = X_test[i]
        reconstructed_flat = predictions[i]

        original_img = original_flat.reshape(image_shape)
        reconstructed_img = reconstructed_flat.reshape(image_shape)

        # Compute per-pixel squared error
        error_map = np.square(original_img - reconstructed_img)

        # Normalize the error map to [0, 1] for heatmap scaling
        norm_error_map = (error_map - np.min(error_map)) / (np.max(error_map) - np.min(error_map) + 1e-8)

        # Create mask for high-error regions (above threshold)
        anomaly_mask = (norm_error_map >= threshold).astype(np.uint8)

        # Convert original grayscale to BGR for overlaying
        original_bgr = cv2.cvtColor((original_img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

        # Create color heatmap
        heatmap_raw = (norm_error_map * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_raw, cv2.COLORMAP_JET)

        # Create final overlay with transparency
        overlay = cv2.addWeighted(original_bgr, 0.6, heatmap_color, 0.4, 0)

        # Optional: draw red where anomaly mask is triggered
        overlay[anomaly_mask == 1] = [255, 0, 0]  # Red for high-error

        # Save and plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 4))
        axes[0].imshow(original_img, cmap="gray")
        axes[0].set_title("Original")
        axes[1].imshow(overlay)
        axes[1].set_title(f"Heatmap (Threshold: {threshold})")

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"anomaly_overlay_{i}.png"))
        plt.close()

        print(f"Saved: anomaly_overlay_{i}.png")

def analyze_mse_distribution(model, X_test, y_test, image_shape, output_dir="analysis_plots"):
    """
    Computes and plots the MSE distribution per image and visualizes per-pixel MSE for individual images.

    Parameters:
        model: Trained Autoencoder.
        X_test: Test images (flattened).
        y_test: Corresponding labels.
        image_shape: Tuple of the original image dimensions (height, width).
        output_dir: Directory to save results.
    """

    os.makedirs(output_dir, exist_ok=True)

    predictions = model.predict([X_test, y_test])

    # Compute per-image MSE
    mse_per_image = np.mean((X_test - predictions) ** 2, axis=1)
    print("Shape of mse_per_image:", mse_per_image.shape)

    # Plot histogram of MSE distribution
    plt.figure(figsize=(8, 5))
    plt.hist(mse_per_image, bins=50, alpha=0.7, color='blue', label="MSE Distribution")
    plt.axvline(np.mean(mse_per_image), color='r', linestyle='dashed', linewidth=2,
                label=f"Mean MSE: {np.mean(mse_per_image):.6f}")
    plt.xlabel("MSE")
    plt.ylabel("Frequency")
    plt.title("MSE Distribution Across Test Images")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "mse_distribution.png"))
    plt.close()

    # Compute per-pixel MSE for each image
    per_pixel_mse = (X_test - predictions) ** 2

    # Plot per-pixel MSE for the first few images
    num_images_to_plot = min(5, X_test.shape[0])
    for i in range(num_images_to_plot):
        mse_image = per_pixel_mse[i].reshape(image_shape)

        plt.figure(figsize=(8, 5))
        plt.imshow(mse_image, cmap='hot', interpolation='nearest')
        plt.colorbar(label='MSE per Pixel')
        plt.title(f"Per-Pixel MSE for Test Image {i}")
        plt.axis('off')

        plt.savefig(os.path.join(output_dir, f"mse_per_pixel_image_{i}.png"))
        plt.close()

    return mse_per_image


def analyze_pixel_validation_loss(model, X_test, y_test, image_shape, output_dir="analysis_plots"):
    """
    Computes and visualizes per-pixel absolute validation loss (difference) per image.

    Parameters:
        model: Trained Autoencoder.
        X_test: Test images (flattened).
        y_test: Corresponding labels.
        image_shape: Tuple of the pooled image dimensions (height, width).
        output_dir: Directory to save results.
    """

    os.makedirs(output_dir, exist_ok=True)

    predictions = model.predict([X_test, y_test])

    # Compute absolute validation loss per image
    abs_loss_per_image = np.mean(np.abs(X_test - predictions), axis=1)
    print("Shape of abs_loss_per_image:", abs_loss_per_image.shape)

    # Plot histogram of absolute validation loss distribution
    plt.figure(figsize=(8, 5))
    plt.hist(abs_loss_per_image, bins=50, alpha=0.7, color='green', label="Absolute Loss Distribution")
    plt.axvline(np.mean(abs_loss_per_image), color='r', linestyle='dashed', linewidth=2,
                label=f"Mean Absolute Loss: {np.mean(abs_loss_per_image):.6f}")
    plt.xlabel("Absolute Validation Loss")
    plt.ylabel("Frequency")
    plt.title("Absolute Validation Loss Distribution Across Test Images")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "abs_validation_loss_distribution.png"))
    plt.close()

    # Compute per-pixel absolute loss for each image
    per_pixel_abs_loss = np.abs(X_test - predictions)

    # Plot per-pixel absolute loss for the first few images
    num_images_to_plot = min(5, X_test.shape[0])
    for i in range(num_images_to_plot):
        loss_image = per_pixel_abs_loss[i].reshape(image_shape)

        plt.figure(figsize=(8, 5))
        plt.imshow(loss_image, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Absolute Loss per Pixel')
        plt.title(f"Per-Pixel Absolute Validation Loss for Test Image {i}")
        plt.axis('off')

        plt.savefig(os.path.join(output_dir, f"abs_loss_per_pixel_image_{i}.png"))
        plt.close()

    return abs_loss_per_image
