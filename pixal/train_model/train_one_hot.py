import os
import sys
import argparse
import logging
import datetime
import yaml
import gc
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# TensorFlow setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
absl.logging.set_stderrthreshold('error')

import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.python.client import device_lib
import tensorflow.keras.backend as K
from numba import cuda

# PIXAL modules
from pixal.modules.config_loader import load_config, resolve_path
from pixal.train_model.autoencoder import Autoencoder

# === Argument Parsing ===
parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="Path to a .npz data file")
parser.add_argument("--config", required=True, help="Path to config YAML")
args = parser.parse_args()

input_file = Path(args.input)
config = load_config(args.config)
path_config = load_config("configs/paths.yaml")

# === TensorFlow/CUDA Settings ===
tf.keras.config.disable_interactive_logging()

if config.enable_memory_growth:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

if config.mixed_precision:
    mixed_precision.set_global_policy('mixed_float16')

if config.TF_GPU_ALLOCATOR:
    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

if config.CPU_MULTI_THREADING:
    tf.config.threading.set_intra_op_parallelism_threads(config.Available_CPU)
    tf.config.threading.set_inter_op_parallelism_threads(config.Available_CPU)

if config.HYBRID_MODE:
    tf.config.set_soft_device_placement(True)
    tf.debugging.set_log_device_placement(True)

# === Clear Previous Session ===
K.clear_session()
gc.collect()
tf.keras.backend.clear_session()
cuda.select_device(0)
cuda.close()

# === Path Setup ===
model_dir = resolve_path(path_config.model_path)
model_dir.mkdir(parents=True, exist_ok=True)

metadata_dir = resolve_path(path_config.metadata_path)
metadata_dir.mkdir(parents=True, exist_ok=True)

log_path = resolve_path(path_config.log_path)
log_path.mkdir(parents=True, exist_ok=True)

log_date = datetime.datetime.now().strftime("%Y-%m-%d")
log_time = datetime.datetime.now().strftime("%H-%M-%S")
log_file = log_path / "train_model.log"

# === Logging Setup ===
logging.basicConfig(
    filename=log_file,
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("pixal")
sys.stdout = open(log_file, 'a')
sys.stderr = sys.stdout
tf.get_logger().setLevel('INFO')
tf.get_logger().addHandler(logging.FileHandler(log_file))

logger.info(f"Devices:\n{device_lib.list_local_devices()}")
logger.info(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
logger.info(f"Num GPUs Available: {len(tf.config.experimental.list_physical_devices('GPU'))}")
logger.info("Script started")

# === Data Loading ===
logger.info(f"Loading data from {input_file}")
dataset = np.load(input_file)
X = dataset["data"] / np.max(dataset["data"])
y = dataset["labels"].astype(np.float32)
original_shape = dataset["shape"]

X = X.reshape(X.shape[0], -1)
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
logger.info(f"Training data shape: {x_train.shape}, Labels: {y_train.shape}")

# === Model Setup ===
model_file = model_dir / f"{config.model_name}.{config.model_file_extension}"
figs_path = resolve_path(path_config.fig_path)
figs_path.mkdir(parents=True, exist_ok=True)

params = {
    'architecture': config.autoencoder_architecture,
    'one-hot_encoding': config.one_hot_encoding,
    'learning_rate': float(config.learning_rate),
    'input_dim': X.shape[1],
    'encoder_names': [f'encoder_layer{i+1}' for i in range(len(config.autoencoder_architecture) - 1)],
    'decoder_names': [f'decoder_layer{i+1}' for i in range(len(config.autoencoder_architecture) - 1)],
    'epochs': config.n_epochs,
    'batchsize': config.batchsize,
    'loss_function': config.loss_function,
    'use_gradient_tape': config.use_gradient_tape,
    'patience': config.patience,
    'modelName': config.model_name,
    'regularization': getattr(config, 'regularization', None),
    'l1_regularization': getattr(config, 'l1_regularization', 0.001),
    'l2_regularization': getattr(config, 'l2_regularization', 0.001),
    'log_date': log_date,
    'timestamp': log_time,
    'fig_path': str(figs_path),
    'model_path': str(model_dir),
    'num_classes': y_train.shape[1],
    'label_latent_size': config.label_latent_size,
    'output_activation': config.output_activation
}

# === Model Training ===
autoencoder = Autoencoder(params)
autoencoder.compile_and_train(x_train, y_train, x_val, y_val, params)

logger.info(f"Saving model to {model_file}")
autoencoder.save_model(model_file)

# === Metadata Save ===
total_time = datetime.datetime.now() - datetime.datetime.strptime(log_time, "%H-%M-%S").replace(
    year=datetime.datetime.now().year,
    month=datetime.datetime.now().month,
    day=datetime.datetime.now().day
)
logger.info("keras config...")
logger.info(f"Model config: {autoencoder.get_config()}")
logger.info(f"Total training time: {total_time}")
params['total_training_time'] = str(total_time)

yaml_path = metadata_dir / f"{config.model_name}.yaml"
with open(yaml_path, 'w') as f:
    yaml.dump(params, f, default_flow_style=False)

logger.info("Script completed")
