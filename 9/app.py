import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
import matplotlib.pyplot as plt  # Importing matplotlib for displaying images

tf.config.experimental.enable_tensor_float_32_execution(False)

# Encoder Network
def build_encoder(latent_dim, input_shape=(28, 28, 1)):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, kernel_size=3, strides=2, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(64, kernel_size=3, strides=2, padding="same", activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation="relu")(x)

    # Latent space
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

    # Sampling layer (reparameterization trick)
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.0)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    z = layers.Lambda(sampling, output_shape=(latent_dim,), name="z")([z_mean, z_log_var])

    encoder = models.Model(inputs, [z_mean, z_log_var, z], name="encoder")
    return encoder

# Decoder Network
def build_decoder(latent_dim, output_shape=(28, 28, 1)):
    latent_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
    x = layers.Reshape((7, 7, 64))(x)
    x = layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding="same", activation="relu")(x)
    outputs = layers.Conv2DTranspose(1, kernel_size=3, padding="same", activation="sigmoid")(x)

    decoder = models.Model(latent_inputs, outputs, name="decoder")
    return decoder

# VAE Model
class VAE(models.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)

        # Add KL divergence loss
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        self.add_loss(K.mean(kl_loss) / np.prod(inputs.shape[1:]))  # Normalize by image size

        return reconstructed

# Define the training function
def train_vae():
    # Hyperparameters
    latent_dim = 2  # Dimensionality of the latent space
    input_shape = (28, 28, 1)  # For the MNIST dataset

    # Build the encoder and decoder
    encoder = build_encoder(latent_dim, input_shape)
    decoder = build_decoder(latent_dim, input_shape)

    # Build the VAE model
    vae = VAE(encoder, decoder)

    # Compile the model
    vae.compile(optimizer='adam', loss='binary_crossentropy')

    # Load the dataset (MNIST in this case)
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
    x_train = np.expand_dims(x_train, -1).astype("float32") / 255
    x_test = np.expand_dims(x_test, -1).astype("float32") / 255

    # Train the VAE
    vae.fit(x_train, x_train, epochs=10, batch_size=128, validation_data=(x_test, x_test))

    # Return trained models
    return vae, encoder, decoder

# Run training
vae, encoder, decoder = train_vae()

# Generate new data
def generate_samples(decoder, n_samples=10):
    # Random points in the latent space
    random_latent_vectors = np.random.normal(size=(n_samples, 2))
    
    # Decode the latent vectors to generate new samples
    generated_images = decoder.predict(random_latent_vectors)
    
    # Display generated images
    for i, img in enumerate(generated_images):
        plt.imshow(img.squeeze(), cmap="gray")  # Corrected plt usage for displaying images
        plt.show()

# Generate and display 10 new samples
generate_samples(decoder)
