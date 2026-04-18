import tensorflow as tf
import os
import time
from .config import *
from .data import get_dataset
from .models import build_generator, build_discriminator
from .utils import save_plot_images, generate_gif

class WGANGP(tf.keras.Model):
    def __init__(self, discriminator, generator, latent_dim, critic_steps, gp_weight=10.0):
        super(WGANGP, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.critic_steps = critic_steps
        self.gp_weight = gp_weight

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(WGANGP, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn
        # Metrics
        self.d_loss_metric = tf.keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = tf.keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def gradient_penalty(self, batch_size, real_images, fake_images):
        """ Calculates the gradient penalty. """
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.discriminator(interpolated, training=True)

        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    @tf.function
    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]

        # Train Discriminator (Critic) multiple times
        for _ in range(self.critic_steps):
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
            with tf.GradientTape() as tape:
                fake_images = self.generator(random_latent_vectors, training=True)
                fake_logits = self.discriminator(fake_images, training=True)
                real_logits = self.discriminator(real_images, training=True)

                d_cost = self.d_loss_fn(real_logits, fake_logits)
                gp = self.gradient_penalty(batch_size, real_images, fake_images)
                d_loss = d_cost + gp * self.gp_weight

            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(zip(d_gradient, self.discriminator.trainable_variables))

        # Train Generator
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            generated_images = self.generator(random_latent_vectors, training=True)
            gen_img_logits = self.discriminator(generated_images, training=True)
            g_loss = self.g_loss_fn(gen_img_logits)

        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))

        # Update metrics
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)

        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
        }

# Loss functions for WGAN-GP
def discriminator_loss(real_logits, fake_logits):
    real_loss = tf.reduce_mean(real_logits)
    fake_loss = tf.reduce_mean(fake_logits)
    return fake_loss - real_loss

def generator_loss(fake_logits):
    return -tf.reduce_mean(fake_logits)

def main():
    print("Loading dataset from Galaxy Zoo...")
    dataset = get_dataset()
    
    print("Building models...")
    generator = build_generator()
    discriminator = build_discriminator()
    
    # Optimizers
    g_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_G, beta_1=BETA_1, beta_2=BETA_2)
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_D, beta_1=BETA_1, beta_2=BETA_2)
    
    # Compile Model
    wgan = WGANGP(discriminator=discriminator, generator=generator, latent_dim=LATENT_DIM, critic_steps=N_CRITIC, gp_weight=GP_WEIGHT)
    wgan.compile(d_optimizer=d_optimizer, g_optimizer=g_optimizer, d_loss_fn=discriminator_loss, g_loss_fn=generator_loss)
    
    # Callbacks
    # TensorBoard
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1)
    
    # Checkpoints
    checkpoint_filepath = os.path.join(CHECKPOINT_DIR, "ckpt-{epoch:02d}.ckpt")
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        save_freq='epoch'
    )
    
    # Custom Callback to save images
    seed = tf.random.normal([16, LATENT_DIM])
    
    class SaveImageCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            predictions = wgan.generator(seed, training=False)
            save_plot_images(predictions, epoch + 1)
            print(f"\nSaved generated images for epoch {epoch + 1}")
            
    print("Starting WGAN-GP training...")
    wgan.fit(dataset, epochs=EPOCHS, callbacks=[tb_callback, model_checkpoint_callback, SaveImageCallback()])

    print("Generating animated GIF of training progress...")
    generate_gif()
    print("Done!")

if __name__ == "__main__":
    main()
