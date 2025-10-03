import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import os

def build_autoencoder(input_dim, latent_dim):
    """Builds and compiles the autoencoder model."""
    # Encoder
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(128, activation='relu')(input_layer)
    encoded = Dense(64, activation='relu')(encoded)
    latent_representation = Dense(latent_dim, activation='relu')(encoded)

    # Decoder
    decoded = Dense(64, activation='relu')(latent_representation)
    decoded = Dense(128, activation='relu')(decoded)
    output_layer = Dense(input_dim, activation='sigmoid')(decoded)

    # Autoencoder model
    autoencoder = Model(input_layer, output_layer)

    # Encoder model
    encoder = Model(input_layer, latent_representation)

    # Decoder model
    encoded_input = Input(shape=(latent_dim,))
    decoder_layers = autoencoder.layers[-3](encoded_input)
    decoder_layers = autoencoder.layers[-2](decoder_layers)
    decoder_layers = autoencoder.layers[-1](decoder_layers)
    decoder = Model(encoded_input, decoder_layers)

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return autoencoder, encoder, decoder

def train_autoencoder(autoencoder, x_train, x_test, epochs=20, batch_size=128):
    """Trains the autoencoder model."""
    print("Training autoencoder...")
    history = autoencoder.fit(x_train, x_train,
                              epochs=epochs,
                              batch_size=batch_size,
                              shuffle=True,
                              validation_data=(x_test, x_test))
    return history

def save_models(encoder, decoder, output_dir="."):
    """Saves the encoder and decoder models."""
    os.makedirs(output_dir, exist_ok=True)
    encoder_path = os.path.join(output_dir, "encoder_model.h5")
    decoder_path = os.path.join(output_dir, "decoder_model.h5")
    encoder.save(encoder_path)
    decoder.save(decoder_path)
    print(f"Encoder model saved to {encoder_path}")
    print(f"Decoder model saved to {decoder_path}")

if __name__ == "__main__":
    print("Building and training autoencoder model...")
    input_dim = 28 * 28
    latent_dim = 32
    # Removed the call to load_and_preprocess_data() as x_train and x_test are already available
    autoencoder, encoder, decoder = build_autoencoder(input_dim, latent_dim)
    autoencoder.summary()
    train_autoencoder(autoencoder, x_train, x_test)
    save_models(encoder, decoder)
    print("Model training and saving complete.")
