import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
absl.logging.set_stderrthreshold('error')

import pixal.modules.plotting as pltm
import numpy as np

stderr_backup = sys.stderr
sys.stderr = open(os.devnull, 'w')
import tensorflow as tf

sys.stderr.close()
sys.stderr = stderr_backup

import matplotlib.pyplot as plt
import cv2



def run():
    # Load test dataset
    dataset = np.load("out_hsvrbg_def.npz")  # Load dataset with X_test and y_test
    #dataset = np.load("val.npz")  # Load dataset with X_test and y_test
    X_test = dataset["data"]
    y_test = dataset["labels"]
    image_shape = dataset["shape"]

    # Normalize and flatten
    X_test = X_test / np.max(X_test)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # Load trained model
    model = tf.keras.models.load_model("models/2025-03-26/rgbhsv_model/testModel.keras")
    
    # Run MSE analysis
    pltm.analyze_mse_distribution(model, X_test, y_test,image_shape,"./figs/rgbhsv_defect_images")
    # Run MSE heatmap visualization
    #pltm.plot_mse_heatmap(model, X_test, y_test)
    
    #check validation loss image
    pltm.analyze_pixel_validation_loss(model, X_test, y_test,image_shape,"./figs/rgbhsv_defect_images")

    # Run MSE heatmap overlay
    pltm.plot_mse_heatmap_overlay(model, X_test, y_test, image_shape, "./figs/rgbhsv_defect_images",threshold=0.7)