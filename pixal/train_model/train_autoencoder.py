import os, sys
import logging
import datetime
import argparse
import yaml
import gc
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from numba import cuda
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.python.client import device_lib
from pixal.modules.config_loader import load_config
from pixal.train_model.autoencoder import Autoencoder

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="Path to a .npz data file")
parser.add_argument("--config", required=True, help="Path to config YAML")
args = parser.parse_args()

# Load config
config = load_config(args.config)

# TensorFlow and CUDA settings
if config.model_training.enable_memory_growth:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

if config.model_training.mixed_precision:
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy('mixed_float16')

if config.model_training.TF_GPU_ALLOCATOR:
    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

if config.model_training.CPU_MULTI_THREADING:
    tf.config.threading.set_intra_op_parallelism_threads(config.model_training.Available_CPU)
    tf.config.threading.set_inter_op_parallelism_threads(config.model_training.Available_CPU)

if config.model_training.HYBRID_MODE:
    tf.config.set_soft_device_placement(True)
    tf.debugging.set_log_device_placement(True)

# Clear session
K.clear_session()
gc.collect()
tf.keras.backend.clear_session()
cuda.select_device(0)
cuda.close()

# Paths
npz_path = Path(args.input)
subdir = npz_path.parent
model_dir = subdir / "model"
metadata_dir = subdir / "metadata"
log_dir = subdir / "logs"

model_dir.mkdir(parents=True, exist_ok=True)
metadata_dir.mkdir(parents=True, exist_ok=True)
log_dir.mkdir(parents=True, exist_ok=True)

# Logging
log_date = datetime.datetime.now().strftime("%Y-%m-%d")
log_time = datetime.datetime.now().strftime("%H-%M-%S")
log_file = log_dir / f"train_model.log"

logger = logging.getLogger("pixal")
logger.setLevel(logging.INFO)
for h in logger.handlers[:]:
    logger.removeHandler(h)
fh = logging.FileHandler(log_file, mode='w')
fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
logger.addHandler(fh)

sys.stdout = open(log_file, 'a')
sys.stderr = sys.stdout
tf.get_logger().setLevel('INFO')
tf.get_logger().addHandler(logging.FileHandler(log_file))

# Load data
logger.info(f"Training model on {npz_path}")
dataset = np.load(npz_path)
X = dataset["data"] / np.max(dataset["data"])
original_shape = dataset["shape"]
X = X.reshape(X.shape[0], -1)
x_train, x_val = train_test_split(X, test_size=0.2, random_state=42)

# Prepare params
params = {
    'architecture': config.model_training.autoencoder_architecture,
    'one_hot_encoding': config.model_training.one_hot_encoding,
    'learning_rate': float(config.model_training.learning_rate),
    'input_dim': X.shape[1],
    'encoder_names': [f'encoder_layer{i+1}' for i in range(len(config.model_training.autoencoder_architecture) - 1)],
    'decoder_names': [f'decoder_layer{i+1}' for i in range(len(config.model_training.autoencoder_architecture) - 1)],
    'epochs': config.model_training.n_epochs,
    'batchsize': config.model_training.batchsize,
    'loss_function': config.model_training.loss_function,
    'use_gradient_tape': config.model_training.use_gradient_tape,
    'patience': config.model_training.patience,
    'modelName': config.model_training.model_name,
    'regularization': getattr(config.model_training, 'regularization', None),
    'l1_regularization': getattr(config.model_training, 'l1_regularization', 0.001),
    'l2_regularization': getattr(config.model_training, 'l2_regularization', 0.001),
    'log_date': log_date,
    'timestamp': log_time,
    'fig_path': str(model_dir),
    'model_path': str(model_dir),
    'label_latent_size': config.model_training.label_latent_size,
    'output_activation': config.model_training.output_activation
}

# Train
autoencoder = Autoencoder(params)
autoencoder = Autoencoder(params)
autoencoder.build_model(input_dim=X.shape[1])

autoencoder.compile_and_train(x_train, x_train, x_val, x_val, params)
model_file = model_dir / f"{config.model_training.model_name}.{config.model_training.model_file_extension}"
autoencoder.save_model(model_file)

# Save metadata
total_time = datetime.datetime.now() - datetime.datetime.strptime(log_time, "%H-%M-%S").replace(
    year=datetime.datetime.now().year,
    month=datetime.datetime.now().month,
    day=datetime.datetime.now().day
)
params['total_training_time'] = str(total_time)
yaml_path = metadata_dir / f"{config.model_training.model_name}.yaml"
with open(yaml_path, 'w') as f:
    yaml.dump(params, f)
logger.info("Training complete")
