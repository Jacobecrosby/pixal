 ```
  ██████╗ ██╗██╗  ██╗ █████╗ ██╗   
  ██╔══██╗██║╚██╗██╔╝██╔══██╗██║     
  ██████╔╝██║ ╚███╔╝ ███████║██║     
  ██╔═══╝ ██║ ██╔██╗ ██╔══██║██║     
  ██║     ██║██╔╝ ██╗██║  ██║███████╗ 
  ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝
   PIXAL – PIXel-based Anomaly Locator
```

PIXAL (PIXel-based Anomaly Locator) is a modular deep learning framework designed for image-based anomaly detection in high-resolution scientific data. Currently applied to identifying defects in detector hardware components for the ATLAS experiment, PIXAL supports training and validation of deep neural networks, with a focus on Autoencoder-based architectures.

The framework includes tools for:

* Image preprocessing, including background removal, alignment, zero-pruning, and ML input processing
* Flexible training with optional one-hot labels and configurable architectures
* Modular validation and anomaly visualization (heatmaps, ROC, loss histograms)
* Metadata tracking and reproducibility for experimental pipelines

PIXAL is highly extensible — other model types and preprocessing pipelines can be added with minimal changes.

# Table of Contents

* [Setup](#setup)
* [Input Data Formatting](#input-data-formatting)
* [Configuration System and Parameters](#configuration-system-and-parameters)
* [Preprocessing Pipeline](#preprocessing-pipeline)
* [Model Training](#model-training)

<a name="setup"></a>
# Setup

PIXAL is tested and works best with **Python 3.10.9**. For consistent results, we recommend creating a clean virtual environment with this version.

### 1. Clone the Repository

```
git clone https://github.com/OSU-HEP-HDL/pixal.git
cd pixal
```

### 2. Setup the Environment
```
source setup.sh
```
This script will:

* Detect your platform (Linux, Windows via WSL or Git Bash, or macOS)

* Create a Python virtual environment in .venv/
 
* Activate the environment

* Install required packages from requirements.txt or requirements-cpu.txt (macOS fallback)

* Set up base configuration files

[!NOTE]
For GPU training, ensure you have a compatible NVIDIA driver and CUDA/cuDNN stack installed. The framework is tested with TensorFlow 2.15+.
[!IMPORTANT]
Note for Windows users: Native Windows is not officially supported. Use WSL2 (Windows Subsystem for Linux) or Git Bash for best results.
[!WARNING]
Note for macOS users: Due to hardware and driver limitations, TensorFlow and related tools will run in CPU-only mode. Training and inference will be slower, but fully functional.

## 3. Verify the Environment

Check to see if the PIXAL framework was properly setup by running the help command.
```
pixal -h
```
<a name="input-data-formatting"></a>
# Input Data Formatting

Since components have different types of images, they should be separated in different directories that are labeled accordingly. The framework parses through nested folders and uses the naming convention for the output.

![Diagram of nested directories for the R0 Triplet Data Flex F1, showing input directory for preprocessing](/pixal/assets/nested_directories.png)

<a name="configuration-system-and-parameters"></a>
# Configuration System and Parameters

PIXAL uses modular YAML-based configuration files to define preprocessing steps, model training parameters, and all path resolutions. This design enables reproducibility, clarity, and easy experimentation.
There are two main configuration files that can be found within the `/configs` folder, they are `parameters.yaml` and `paths.yaml`. 

## Parameters

The `parameters.yaml` file contains all high-level control flags. The file is split into three sections, `preprocessing`, `model_training`, and `plotting`. 

### Preprocessing

Defines how images are cleaned and transformed:

* remove_background: Max workers are the number of threads for parallel processing when removing backgrounds from the images.
* alignment: parameters for KNN and RANSAC-based image alignment. Includes addtional metric and image flags. 
* preprocessor: controls pooling, zero pruning, color channels, and .npz output.
* rename_images: optionally renames images to folder-consistent names.

### Model Training

Covers everything needed to build and train the neural network:

* Memory handling: GPU/CPU flags, threading, memory growth, and hybrid options.
* Architecture: latent layer size, encoder/decoder depth, label encoding, one-hot encoding flag.
* Training control: batch size, learning rate, optimizer settings, loss functions.
* Regularization: supports l1, l2, or combined with tunable coefficients.
* Early stopping: using patience and min_delta.

### Plotting

Choose what diagnostic plots to generate after training:

* ROC/Recall, pixel-wise MSE/MAE, distribution comparisons, confusion matrix, etc.
* Log-based vs absolute loss plotting.
* Loss cut threshold to define anomaly threshold

## Paths

PIXAL resolves all data inputs/outputs relative to a few base directories. There are two main base paths, all preprocessing and model trainings are output to `/out` and all validation and detection are output to `/validate`. This YAML allows centralized control of:

* `component_model_path`: where trained models and logs are saved.
* `component_validate_path`: path used during validation and detection.

The naming of these two sections are the only names the user should alter. Each section (like remove_background_path, aligned_images_path, etc.) defines a name and a base, which are combined at runtime using PIXAL’s recursive path resolution system.

### Example

```
aligned_images_path:
  aligned_images: "aligned_images"
  base: *preprocessed_images_path
```
This lets PIXAL dynamically build:
```
out/R0_Triplet_Data_Flex_F1_pink_prune_2pool_rgb/preprocessed_images/aligned_images
```

### Advanced Behavior

* Hierarchical Namespacing: All configurations are parsed into nested Python namespaces (`config.preprocessing.preprocessor.pool_size`, etc.) for intuitive access.

* Metadata: PIXAL automatically stores and saves parameters, including bounding box crop data from zero-pruning as metadata for use in validation.

* Multi-file Merging:  PIXAL merges multiple metadata YAMLs in a directory into one logical config object. These merged multiple YAMLs in a directory into one logical config object. This gives users separate reusable preprocessing.yaml, model_training.yaml, and plotting.yaml files while still combining them at runtime.

<a name="preprocessing-pipeline"></a>
# Preprocessing Pipeline

PIXAL includes a modular and efficient preprocessing pipeline designed to prepare image data for machine learning-based anomaly detection. The image shown is the front of the R0 Triplet Data Flex Flavor 1 which will be used as an example going through this pipeline, taken by a Tagarno Microscope. Below are the key stages:

<p align="center">
  <img src="/pixal/assets/preprocessing/image_436995.jpg" alt="R0 Triplet Data Flex Flavor 1 front with no preprocessing"/>
</p>

## Background Removal

Removes the background from each input image to isolate the object of interest. This is done using the `rembg` library with optional multithreaded support.

Purpose:
Reduce noise and standardize input for feature extraction.

**Config settings:**
```
preprocessing:
  remove_background:
    max_workers: 8
  rename_images: true
```
**Output:**
`component/preprocessed_images/background_removed/`

<p align="center">
  <img src="/pixal/assets/preprocessing/R0_Triplet_Data_Flex_F1_F_Pink_bg_000_no_bg.png" alt="R0 Triplet Data Flex Flavor 1 front with its background removed"/>
</p>

## Image Alignment

Aligns each background-removed image to a reference using feature matching (KNN, RANSAC). Ensures consistent orientation and spatial scale.

**Purpose:**
Standardize object placement across the dataset.

**Config settings:**
```
preprocessing:
  alignment:
    knn_ratio: 0.8
    number_of_points: 5
    ransac_threshold: 7.0
    MIN_SCORE_THRESHOLD: 0.5
    MAX_MSE_THRESHOLD: 10.0
    MIN_GOOD_MATCHES: 20
  draw_matches: true
  save_metrics: true
  save_overlays: true
```
**Output:**
`preprocessed_images/aligned_images/`
`figures/aligned_metrics/`

<p align="center">
  <img src="/pixal/assets/preprocessing/match_2_0.png" alt="Two R0 Triplet Data Flexes Flavor 1 showing 10 matching points found using KNN"/>
</p>

## Zero Pruning (Optional)

Cropping step that removes zero-valued background pixels after alignment. The system finds the tightest bounding box around the non-zero pixels (with configurable padding) and crops all images to the same region.

**Purpose:**
Reduce input dimensionality while preserving relevant information.

**Config settings:**
```
preprocessing:
  preprocessor:
    zero_pruning: true
    zero_pruning_padding: 5
```
**Output**
Internally processed images; cropping dimensions are saved in:
`metadata/preprocessing.yaml`

<p align="center">
  <img src="/pixal/assets/preprocessing/crop_preview.png" alt="R0 Triplet Data Flexes Flavor 1 after zero pruning"/>
</p>

## Preprocesor -> ML Input Conversion

Converts aligned (and optionally pruned) images into normalized ML-ready inputs. This includes:

* Channel selection can be any combination of (R, G, B, H, S, V)
* Average pooling to reduce resolution
* Per-channel normalization
* .npz output containing data, labels (if applicable), and shape

```
preprocessing:
  preprocessor:
    file_name: "out.npz"
    pool_size: 2
    channels: ["R", "G", "B"]
```
**Output:**
`out/<component>/<type>/out.npz`

## Metadata Output
Important parameters like `crop_box`, `input_dim`, and processing shapes are saved to:
`out/<component>/<type>/metadata/preprocessing.yaml`

<a name="model-training"></a>
# Model Training

PIXAL supports flexible and modular training of deep learning models (currently autoencoders) for anomaly detection in pixel-aligned image data.

## The Autoencoder Architecture

An Autoencoder is a type of neural network that learns to compress and reconstruct its input. It's structured into three parts:

* Encoder: Compresses the input image into a smaller latent representation. This part captures the most essential features of the data.
* Latent Space: The compressed representation. It’s the "bottleneck" that forces the network to learn meaningful features.
* Decoder: Attempts to reconstruct the original image from the latent representation.

In the context of PIXAL, this model learns to reproduce defect-free components. During validation, poor reconstruction (i.e., higher pixel-wise loss) indicates anomalous or defective regions.

<p align="center">
  <img src="/pixal/assets/autoencoder_architecture.png" alt="R0 Triplet Data Flexes Flavor 1 after zero pruning"/>
</p>

## Input Format

Before training, images must be preprocessed and converted into .npz files using the preprocessing pipeline (see previous section). Each .npz file contains:

* `data`: flattened, normalized image vectors
* `labels`: (only if using one-hot encoding)
* `shape`: original image shape post-pooling or zero-pruning

## Training Modes

PIXAL supports **two training modes:**

### 1. Per-Type Model (default)

Trains a separate model for each image type (e.g. component variant or class). Each .npz file corresponds to a single type.
```
model_training:
  one_hot_encoding: False
```
* **Benefits:** Higher performance, more specific models

* **Model Output:**
The model is saved both as a `.keras` file and its weights as `<model_name>.weights.h5`, these can be found in:
`out/<component>/<type>/model/<model_name>.weights.h5`
Currently, models are loaded and rebuilt using the `<model_name>.weights.h5` for validation.

### 2. One-Hot Encoding Mode

Trains a single model on all types of images, with one-hot encoded class labels appended to the latent space.
```
model_training:
  one_hot_encoding: True
```
* **Benefits:** Generalized model across types
* **Model Output:**
Just as the per-type mode, the model is saved both as a `.keras` file and its weights as `<model_name>.weights.h5`, these can be found in:
`out/<component>/model/<model_name>.weights.h5`
Currently, models are loaded and rebuilt using the `<model_name>.weights.h5` for validation.

# Validation and Detection

