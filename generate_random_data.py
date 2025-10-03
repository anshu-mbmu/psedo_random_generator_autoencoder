import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

def load_decoder_model(model_path="decoder_model.h5"):
    """Loads the trained decoder model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Decoder model not found at {model_path}. Please train the model first.")
    decoder = tf.keras.models.load_model(model_path)
    return decoder

def generate_pseudo_random_data(decoder, latent_dim, num_samples=100):
    """Generates pseudo-random data using the decoder."""
    # Generate random points in the latent space
    random_latent_vectors = np.random.normal(size=(num_samples, latent_dim))

    # Decode the random latent vectors to generate pseudo-random data
    generated_data = decoder.predict(random_latent_vectors)

    return generated_data

def visualize_generated_data(generated_data, num_display=10):
    """Visualizes a subset of the generated data (assuming image data)."""
    # Reshape the generated data to image format (for visualization)
    generated_images = generated_data.reshape(generated_data.shape[0], 28, 28)

    # Display the generated pseudo-random images
    plt.figure(figsize=(num_display, 2))
    for i in range(min(num_display, generated_images.shape[0])):
        ax = plt.subplot(1, num_display, i + 1)
        plt.imshow(generated_images[i].reshape(28, 28), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.suptitle("Generated Pseudo-random Images")
    plt.show()

if __name__ == "__main__":
    print("Generating pseudo-random data...")
    latent_dim = 32 # Needs to match the latent_dim used in training
    try:
        decoder = load_decoder_model()
        generated_data = generate_pseudo_random_data(decoder, latent_dim, num_samples=25)
        print(f"Shape of generated_data: {generated_data.shape}")
        visualize_generated_data(generated_data, num_display=10)
        print("Pseudo-random data generation complete.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run autoencoder_model.py first to train and save the decoder model.")
