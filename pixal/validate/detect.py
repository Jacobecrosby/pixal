import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
absl.logging.set_stderrthreshold('error')

import pixal.modules.plotting as pltm
import pixal.train_model.autoencoder as autoencoder
import numpy as np
import logging
from pathlib import Path

stderr_backup = sys.stderr
sys.stderr = open(os.devnull, 'w')
import tensorflow as tf

sys.stderr.close()
sys.stderr = stderr_backup

import matplotlib.pyplot as plt
import cv2

logger = logging.getLogger("pixal")

def run_detection(dataset, model_dir, metric_dir, config=None, quiet=False):
    # Load dataset with X_test and y_test
    X_test = dataset["data"]
    y_test = dataset["labels"]
    image_shape = dataset["shape"]

    # Flatten
    X_test = X_test.reshape(X_test.shape[0], -1)
    y_test = y_test.reshape(y_test.shape[0], -1)
   
    # Load trained model
    model = (model_dir / f"{config.model_name}.{config.model_file_extension}")
    model = autoencoder.Autoencoder.load_model(model)
    #model = tf.keras.models.load_model(model)
    
    # plot prediction distribution
    pltm.plot_prediction_distribution(model, X_test, y_test, metric_dir / "validation")

    pltm.plot_truth_distribution(X_test, metric_dir / "validation")

    # Run MSE analysis
    pltm.analyze_mse_distribution(model, X_test, y_test, image_shape, metric_dir / "validation")
    # Run MSE heatmap visualization
    #pltm.plot_mse_heatmap(model, X_test, y_test)
    
    #check validation loss image
    pltm.analyze_pixel_validation_loss(model, X_test, y_test,image_shape, metric_dir / "validation")

    # Run MSE heatmap overlay
    pltm.plot_mse_heatmap_overlay(model, X_test, y_test, image_shape, metric_dir / "validation",threshold=0.7)

    # Run pixel-wise predictions
    pltm.plot_anomaly_detection_curves(model,X_test,y_test, '', metric_dir / "validation")

    # Run pixel-wise predictions
    pltm.plot_pixel_predictions(model,X_test,y_test, "Pixel-wise Prediction Accuracy",metric_dir / "validation")


def run(npz_dir, model_dir, metric_dir, config=None, quiet=False):
    # Load test dataset
    file_name = config.preprocessor.file_name if config and hasattr(config.preprocessor, 'file_name') else "out.npz"
    #model_name = config.model.name if config and hasattr(config.model, 'name') else "testModel.keras"
    npz_dir = Path(npz_dir)

    npz = npz_dir / file_name
    dataset = np.load(npz)  

    run_detection(dataset, model_dir, metric_dir, config=config, quiet=quiet)
    