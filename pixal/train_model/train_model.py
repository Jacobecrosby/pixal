import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
absl.logging.set_stderrthreshold('error')

stderr_backup = sys.stderr
sys.stderr = open(os.devnull, 'w')

import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.keras import mixed_precision

sys.stderr.close()
sys.stderr = stderr_backup

import numpy as np
import logging
import argparse
import tensorflow as tf
import datetime
import sys
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
from glob import glob

from numba import cuda
import gc
from pixal.modules.config_loader import load_config
from pixal.train_model.autoencoder import Autoencoder
from pixal.modules.config_loader import load_config, resolve_path
from pixal.modules.model_training import masked_mse_loss


def run(input_file, config, quiet):
    tf.keras.config.disable_interactive_logging()
    
    path_config = load_config("configs/paths.yaml")
   
    # Clear GPU
    gc.collect()
    tf.keras.backend.clear_session()
    cuda.select_device(0)
    cuda.close()

    # TensorFlow setup
    if config.enable_memory_growth:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logging.info("Memory growth enabled")

    if config.mixed_precision:
        mixed_precision.set_global_policy('mixed_float16')
        logging.info("Mixed precision enabled")

    if config.TF_GPU_ALLOCATOR:
        os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
        logging.info("Using cuda_malloc_async")

    if config.CPU_MULTI_THREADING:
        tf.config.threading.set_intra_op_parallelism_threads(config.Available_CPU)
        tf.config.threading.set_inter_op_parallelism_threads(config.Available_CPU)
        logging.info("Configured TensorFlow to use multiple CPU threads")

    if config.HYBRID_MODE:
        tf.config.set_soft_device_placement(True)
        tf.debugging.set_log_device_placement(True)
        logging.info("Hybrid CPU/GPU mode enabled")

    # Load training data
    logging.info(f"Loading data from {input_file}")
    
    if config.one_hot_encoding:
         
        model_dir = resolve_path(path_config.model_path)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        if not input_file:
            input_file = resolve_path(path_config.component_model_path) / config.preprocessor.file_name
            input_file = Path(input_file)

        metadata_dir = resolve_path(path_config.metadata_path)
        metadata_dir.mkdir(parents=True, exist_ok=True)

        # Set up logging
        log_date = datetime.datetime.now().strftime("%Y-%m-%d")
        log_time = datetime.datetime.now().strftime("%H-%M-%S")
        
        log_path = resolve_path(path_config.log_path)
        log_path.mkdir(parents=True, exist_ok=True)
        log_file = Path(log_path / "train_model.log")

        logging.basicConfig(
            filename= log_file,
            filemode="w",
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        logger = logging.getLogger("pixal")
        
        sys.stdout = open(log_file, 'a')
        sys.stderr = sys.stdout
        tf.get_logger().setLevel('INFO')
        tf.get_logger().addHandler(logging.FileHandler(log_file))
         # Device logging

        logging.info(device_lib.list_local_devices())
        logging.info(tf.test.is_built_with_cuda())
        logging.info(f"Num GPUs Available: {len(tf.config.experimental.list_physical_devices('GPU'))}")

        logging.info("Script started")

        dataset = np.load(input_file)
        X = dataset["data"] / np.max(dataset["data"])
        y = dataset["labels"].astype(np.float32)
        original_shape = dataset["shape"]

        X = X.reshape(X.shape[0], -1)
        x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        logging.info(f"Training data shape: {x_train.shape}, Labels: {y_train.shape}")

        input_dim = X.shape[1]


        model_file = os.path.join(model_dir, config.model_name + "." + config.model_file_extension)
        figs_path = resolve_path(path_config.fig_path)
        figs_path.mkdir(parents=True, exist_ok=True)
        
        #if config.loss_function == 'masked_mse':
        #    loss_fn = masked_mse_loss
        #else:
        #    loss_fn = config.loss_function  # e.g., 'mse'

        params = {
            'architecture': config.autoencoder_architecture,
            'one-hot_encoding': config.one_hot_encoding,
            'learning_rate': float(config.learning_rate),
            'input_dim': input_dim,
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

        autoencoder = Autoencoder(params)
        autoencoder.compile_and_train(x_train, y_train, x_val, y_val, params)

        logging.info(f"Saving model to {model_file}")
        autoencoder.save_model(model_file)

        total_time = datetime.datetime.now() - datetime.datetime.strptime(log_time, "%H-%M-%S").replace(
            year=datetime.datetime.now().year,
            month=datetime.datetime.now().month,
            day=datetime.datetime.now().day
        )
        logging.info("keras config...")
        logging.info(f"Model config: {autoencoder.get_config()}")
        logging.info(f"Total training time: {total_time}")
        logging.info(f"Saving parameters to {metadata_dir}")
        params['total_training_time'] = str(total_time)
        yaml_path = os.path.join(metadata_dir, config.model_name + ".yaml")
        with open(yaml_path, 'w') as file:
            yaml.dump(params, file, default_flow_style=False)
        logging.info("Script completed")
    
    else:
        if not input_file:
            input_file = resolve_path(path_config.component_model_path)
            #input_file = Path(input_file)

        input_folder = Path(input_file)
        npz_files = glob(str(input_folder / "*/*.npz"))

        for npz_path in npz_files:
            npz_path = Path(npz_path)
            dataset = np.load(npz_path)
            X = dataset["data"] / np.max(dataset["data"])
            original_shape = dataset["shape"]
            X = X.reshape(X.shape[0], -1)
            x_train, x_val = train_test_split(X, test_size=0.2, random_state=42)

            # === New: Setup per-subdirectory paths ===
            print(f"Processing {npz_path}")
            print("Subdir:", npz_path.parent)
            subdir = npz_path.parent
            model_dir = subdir / "model"
            metadata_dir = subdir / "metadata"
            log_dir = subdir / "logs"

            model_dir.mkdir(parents=True, exist_ok=True)
            metadata_dir.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)

            log_date = datetime.datetime.now().strftime("%Y-%m-%d")
            log_time = datetime.datetime.now().strftime("%H-%M-%S")
            log_file = log_dir / f"train_model.log"

            logging.basicConfig(
                filename=log_file,
                filemode="w",
                level=logging.INFO,
                format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            )

            logging.info(device_lib.list_local_devices())
            logging.info(tf.test.is_built_with_cuda())
            logging.info(f"Num GPUs Available: {len(tf.config.experimental.list_physical_devices('GPU'))}")

            logging.info("Script started")
            logger = logging.getLogger("pixal")
            sys.stdout = open(log_file, 'a')
            sys.stderr = sys.stdout
            tf.get_logger().setLevel('INFO')
            tf.get_logger().addHandler(logging.FileHandler(log_file))

            logging.info("Script started")

            model_file = model_dir / f"{config.model_name}.{config.model_file_extension}"

            params = {
                'architecture': config.autoencoder_architecture,
                'one_hot_encoding': config.one_hot_encoding,
                'learning_rate': float(config.learning_rate),
                'input_dim': X.shape[1],
                'encoder_names': [f'encoder_layer{i+1}' for i in range(len(config.autoencoder_architecture) - 1)],
                'decoder_names': [f'decoder_layer{i+1}' for i in range(len(config.autoencoder_architecture) - 1)],
                'epochs': config.n_epochs,
                'batchsize': config.batchsize,
                'loss_function': config.loss_function,
                'use_gradient_tape': config.use_gradient_tape,
                'patience': config.patience,
                'modelName': model_file.stem,
                'regularization': getattr(config, 'regularization', None),
                'l1_regularization': getattr(config, 'l1_regularization', 0.001),
                'l2_regularization': getattr(config, 'l2_regularization', 0.001),
                'log_date': log_date,
                'timestamp': log_time,
                'fig_path': str(model_dir),
                'model_path': str(model_dir),
                'label_latent_size': config.label_latent_size,
                'output_activation': config.output_activation
            }

            autoencoder = Autoencoder(params)
            autoencoder.compile_and_train(x_train, x_train, x_val, x_val, params)
            autoencoder.save_model(model_file)

            total_time = datetime.datetime.now() - datetime.datetime.strptime(log_time, "%H-%M-%S").replace(
                year=datetime.datetime.now().year,
                month=datetime.datetime.now().month,
                day=datetime.datetime.now().day
            )

            logging.info("keras config...")
            logging.info(f"Model config: {autoencoder.get_config()}")
            logging.info(f"Total training time: {total_time}")
            logging.info(f"Saving parameters to {metadata_dir}")

            params['total_training_time'] = str(total_time)
            yaml_path = metadata_dir / f"{config.model_name}.yaml"
            with open(yaml_path, 'w') as file:
                yaml.dump(params, file, default_flow_style=False)

            logging.info("Script completed")