import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Define constants
NOISE_DIM = 100  # Dimension of the random noise input for the generator
IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = 28, 28, 1  # Image dimensions (using grayscale images here)
BATCH_SIZE = 128  # Batch size for training
EPOCHS = 50000  # Number of epochs for training
tf.config.experimental.enable_tensor_float_32_execution(False)

# Load and preprocess the dataset (use MNIST here for simplicity)
def load_dataset():
    (x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()  # MNIST dataset

    # Normalize the data between -1 and 1 for the generator output (tanh activation)
    x_train = (x_train - 127.5) / 127.5  
    x_train = np.expand_dims(x_train, axis=-1)  # Expand dimensions to include channel info

    return x_train

# Generator model
def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(256, input_dim=NOISE_DIM),
        layers.LeakyReLU(alpha=0.2),
        layers.BatchNormalization(),
        layers.Dense(512),
        layers.LeakyReLU(alpha=0.2),
        layers.BatchNormalization(),
        layers.Dense(1024),
        layers.LeakyReLU(alpha=0.2),
        layers.BatchNormalization(),
        layers.Dense(IMG_HEIGHT * IMG_WIDTH * IMG_CHANNELS, activation='tanh'),
        layers.Reshape((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))  # Reshape into an image
    ])
    
    return model

# Discriminator model
def build_discriminator():
    model = tf.keras.Sequential([
        layers.Flatten(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)),
        layers.Dense(512),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(256),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(1, activation='sigmoid')  # Binary classification (real/fake)
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5), 
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

# GAN model (combining generator and discriminator)
def build_gan(generator, discriminator):
    discriminator.trainable = False  # Freeze discriminator's weights when training the GAN
    
    model = tf.keras.Sequential([
        generator,
        discriminator
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5), 
                  loss='binary_crossentropy')
    return model

# Function to generate random noise
def generate_random_noise(batch_size, noise_dim):
    return np.random.normal(0, 1, (batch_size, noise_dim))

# Function to visualize generated images
def save_generated_images(epoch, generator, examples=25, dim=(5, 5), figsize=(5, 5)):
    noise = generate_random_noise(examples, NOISE_DIM)
    generated_images = generator.predict(noise)
    generated_images = (generated_images + 1) / 2.0  # Rescale from [-1, 1] to [0, 1]

    plt.figure(figsize=figsize)
    for i in range(examples):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(generated_images[i, :, :, 0], cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"gan_generated_image_epoch_{epoch}.png")
    plt.close()

# Train the GAN
def train_gan(generator, discriminator, gan, dataset, epochs, batch_size):
    real = np.ones((batch_size, 1))  # Labels for real images
    fake = np.zeros((batch_size, 1))  # Labels for fake images
    
    for epoch in range(epochs):
        # 1. Train the discriminator
        # Select a random batch of real images
        idx = np.random.randint(0, dataset.shape[0], batch_size)
        real_images = dataset[idx]

        # Generate fake images from random noise
        noise = generate_random_noise(batch_size, NOISE_DIM)
        fake_images = generator.predict(noise)

        # Train the discriminator (real classified as 1 and fake as 0)
        d_loss_real = discriminator.train_on_batch(real_images, real)
        d_loss_fake = discriminator.train_on_batch(fake_images, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # 2. Train the generator via the GAN (generator wants discriminator to classify generated images as real)
        noise = generate_random_noise(batch_size, NOISE_DIM)
        g_loss = gan.train_on_batch(noise, real)  # The generator wants to fool the discriminator

        # Print the progress
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}/{epochs} [D loss: {d_loss[0]}, acc: {100*d_loss[1]}] [G loss: {g_loss}]")
            
            # Save generated images at every 1000 epochs
            save_generated_images(epoch, generator)

# Main function to set up and train the GAN
def main():
    dataset = load_dataset()

    # Build and compile the models
    generator = build_generator()
    discriminator = build_discriminator()
    gan = build_gan(generator, discriminator)

    # Train the GAN
    train_gan(generator, discriminator, gan, dataset, EPOCHS, BATCH_SIZE)

if __name__ == "__main__":
    main()
