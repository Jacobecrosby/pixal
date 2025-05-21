import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
absl.logging.set_stderrthreshold('error')

stderr_backup = sys.stderr
sys.stderr = open(os.devnull, 'w')

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.callbacks import EarlyStopping

sys.stderr.close()
sys.stderr = stderr_backup

import numpy as np
import yaml
import logging
import matplotlib.pyplot as plt
import json



class Autoencoder(tf.keras.Model):

    def __init__(self, params, **kwargs):
        super(Autoencoder, self).__init__(**kwargs)

        self.params = params
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
        input_dim = params['input_dim']
        latent_dim = params['architecture'][-1]

        self.logger.info("Initializing Autoencoder model...")
        self.logger.info(f"Autoencoder model architecture: {params['architecture']}")

        # Concatenate image data and labels at input
        #self.concat_input = Concatenate()([self.input_img, self.input_label])

        # Build encoder layers
        self.encoder = tf.keras.Sequential(name="encoder")
        encoder_arch = params['architecture'][:-1]

        self.encoder.add(tf.keras.layers.Input(shape=(input_dim,)))  # Flatten the input dimension
        
        for i, units in enumerate(encoder_arch):
            self.encoder.add(tf.keras.layers.Dense(units, 
                                                   activation=tf.nn.leaky_relu, # if alpha needs to be changed: activation=lambda x: tf.nn.leaky_relu(x, alpha=0.1))
                                                   activity_regularizer=tf.keras.regularizers.l2(params['l2_regularization']),
                                                   name=params['encoder_names'][i]))

        # Latent space
        self.latent_layer = tf.keras.layers.Dense(latent_dim, activation=tf.nn.leaky_relu, name="latent")

        # Project labels into a higher-dimensional space
        self.label_projection = Dense(params['label_latent_size'], activation='relu', name="label_transform")

        # Build decoder layers
        decoder_arch = encoder_arch[::-1]
        self.decoder = tf.keras.Sequential(name="decoder")
        self.decoder.add(Input(shape=(latent_dim+params['label_latent_size'],)))  # Explicit input for decoder. Adds label latent size to account for concatenated latent space and transformed labels

        for i, units in enumerate(decoder_arch):
            self.decoder.add(tf.keras.layers.Dense(units, 
                                                   activation=tf.nn.leaky_relu, 
                                                   activity_regularizer=tf.keras.regularizers.l2(params['l2_regularization']), 
                                                   name=params['decoder_names'][i]))

        # Output layer
        #self.output_layer = tf.keras.layers.Dense(input_dim, activation=tf.nn.leaky_relu, name="output") # May need sigmoid for activation
        self.output_layer = tf.keras.layers.Dense(input_dim, activation=params['output_activation'], name="output") # tf.nn.leaky_relu or sigmoid
        self.logger.info("Autoencoder model initialized successfully.")

    def call(self, inputs):
        self.logger.debug("Starting forward pass...")
        
        x, labels = inputs  # Unpack input tuple (image, labels)
        
        # Concatenate labels into the encoder input
        #x = tf.concat([x, labels], axis=1)

        # 🔹 Encode Image
        encoded = self.encoder(x)
        latent = self.latent_layer(encoded)

        # 🔹 Transform Labels (Trainable Projection)
        transformed_labels = self.label_projection(labels)

        # 🔹 Concatenate transformed labels with latent space
        latent_combined = tf.concat([latent, transformed_labels], axis=1)

        # 🔹 Decode
        decoded = self.decoder(latent_combined)
        
        self.logger.debug("Forward pass complete.")
        return self.output_layer(decoded)

    def get_config(self):
        config = super().get_config()
        config.update({
            'params': self.params
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        params = config.pop('params')
        return cls(params, **config)

    def compile_and_train(self, x_train, y_train, x_val, y_val, params):
        # Early stopping and checkpointing
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=params['patience'],
            mode='min',
            verbose=1,
            restore_best_weights=True
        )

        checkpoint = ModelCheckpoint(
            filepath=f"{params['model_path']}/{params['modelName']}.keras", 
            verbose=1, 
            save_freq='epoch'
        )

        # Compile the model
        self.logger.info("Compiling the model...")
        self.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
                    loss=params['loss_function'],
                    metrics=['mse']) # Mean Squared Error (measures reconstruction quality)
        
        if params['use_gradient_tape']:
            self.logger.info("Training with gradient tape...")

            for epoch in range(params['epochs']):
                for step, (x_batch, y_batch) in enumerate(zip(x_train, y_train)):  # Include labels
                    with tf.GradientTape() as tape:
                        predictions = self.call(x_batch, y_batch)  # Pass both image & label
                        loss = self.compiled_loss(x_batch, predictions)  # MSE loss
                    gradients = tape.gradient(loss, self.trainable_variables)
                    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

                    if step % 10 == 0:
                        self.logger.info(f"Epoch {epoch + 1}, Step {step}, Loss: {loss.numpy():.4f}")
        else:
            # Train the model with default fit method
            self.logger.info("Training the model with standard fit method...")
            #Args: training data, target data, batch_size, epochs, verbose, callbacks, validation_data=[x_val, x_target], val_batch_size
            # Train the model using (x_train, y_train) as input
            history = self.fit(
                [x_train, y_train],  # Input: (image data + labels)
                x_train,  # Target is the same as input (autoencoder behavior)
                batch_size=params['batchsize'],
                epochs=params['epochs'],
                verbose=1,
                callbacks=[early_stopping, checkpoint],
                validation_data=([x_val, y_val], x_val),
                validation_batch_size=params['batchsize']
        ) 
        # Save loss values to a file
        loss_history = {
            "train_loss": history.history["loss"],
            "val_loss": history.history["val_loss"]
            }
        save_dir = params['model_path']
        logging.info("save_dir: {save_dir}")
        os.makedirs(save_dir, exist_ok=True)  # Creates the directory if it does not exist
        loss_file_path = os.path.join(save_dir, "loss_history.json")
        with open(loss_file_path, "w") as f:
            json.dump(loss_history, f)

        # Plot and save the loss curve
        self.plot_loss(history.history, params['fig_path'])

    def plot_loss(self, history, save_dir):
        """Plot and save training vs validation loss"""
        plt.figure(figsize=(8, 5))
        plt.plot(history['loss'], label="Train Loss")
        plt.plot(history['val_loss'], label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.legend()
        plt.grid()
        
        loss_plot_path = os.path.join(save_dir, "loss_plot.png")
        plt.savefig(loss_plot_path)
        plt.close()
        
        logging.info(f"Loss plot saved to {loss_plot_path}")

    def evaluate_model(self, test_data, test_labels):
        # Evaluate the model on test data
        return self.evaluate(test_data, test_labels)

    def predict_model(self, new_data):
        # Generate predictions
        return self.predict(new_data)
    
    def save_model(self, save_path):
        """Save the model to the specified path."""
        self.save(save_path)
        self.logger.info(f"Model saved to {save_path}")
      
    @classmethod
    def load_model(cls, load_path):
        """Load a saved model from the specified path."""
        return tf.keras.models.load_model(load_path, custom_objects={'Autoencoder': cls})
