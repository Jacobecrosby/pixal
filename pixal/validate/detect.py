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
    
    predictions = model.predict([X_test, y_test])
    
    if config.plotting.plot_distributions:
        # plot prediction distribution
        pltm.plot_prediction_distribution(predictions, metric_dir / "validation")
    
    if config.plotting.plot_distributions:
        # plot truth distribution
        pltm.plot_truth_distribution(X_test, metric_dir / "validation")

    if config.plotting.plot_distributions:
        pltm.plot_combined_distribution(X_test, predictions, metric_dir / "validation")

    # Run MSE analysis
    #pltm.analyze_mse_distribution(X_test, predictions, image_shape, metric_dir / "validation")
    # Run MSE heatmap visualization
    #pltm.plot_mse_heatmap(model, X_test, y_test)
    
    #check validation loss image
    #pltm.analyze_pixel_validation_loss(X_test, predictions, image_shape, metric_dir / "validation")
    
    if config.plotting.plot_anomaly_heatmap:
        # Run MSE heatmap overlay
        pltm.plot_mse_heatmap_overlay(X_test, predictions, image_shape, metric_dir / "validation",threshold=config.loss_cut, use_log_threshold=config.use_log_loss)
    
    if config.plotting.plot_roc_recall_curve:
        # Run pixel-wise predictions
        pltm.plot_anomaly_detection_curves(X_test, predictions, '', metric_dir / "validation")
    
    if config.plotting.plot_pixel_predictions:
        # Run pixel-wise predictions
        pltm.plot_pixel_predictions(X_test, predictions, "Pixel-wise Prediction Accuracy",metric_dir / "validation")
   
    if config.plotting.plot_confusion_matrix:
        # Run confusion matrix
        pltm.plot_confusion_matrix(X_test, predictions, metric_dir / "validation")

    if config.plotting.plot_loss:
        # Run loss analysis
        pltm.plot_pixel_loss_and_log_loss(X_test, predictions, metric_dir / "validation", loss_threshold=config.loss_cut)
        pltm.plot_channelwise_pixel_loss(X_test, predictions, config, metric_dir / "validation", loss_threshold=config.loss_cut)

def run(npz_dir, model_dir, metric_dir, config=None, quiet=False):
    # Load test dataset
    file_name = config.preprocessor.file_name if config and hasattr(config.preprocessor, 'file_name') else "out.npz"
    #model_name = config.model.name if config and hasattr(config.model, 'name') else "testModel.keras"
    npz_dir = Path(npz_dir)

    npz = npz_dir / file_name
    dataset = np.load(npz)  

    run_detection(dataset, model_dir, metric_dir, config=config, quiet=quiet)
    