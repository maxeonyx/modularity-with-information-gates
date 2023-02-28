"""
Tensorflow implementation of an auto-encoder for MNIST-compatible datasets
"""

from einml.prelude import *

class AE(Model):

    @u.tf_scope
    def __init__(
        self,
        latent_dim = 2,
        name=None
    ):
        super().__init__(name=name)

        self.latent_dim = latent_dim

        inp = Input((28, 28, 1))

        # encoder

        x = inp
        x = layers.Conv2D(16, 5, strides=(2, 2), padding='same')(x)
        x = layers.LeakyReLU()(x)
        x = layers.Conv2D(32, 5, strides=(2, 2), padding='same')(x)
        x = layers.LeakyReLU()(x)
        x = layers.Flatten()(x)
        x = Dense(2)(x)

        self.encoder = tf.keras.Model(inp, x)

        self.encoder.summary()

        # decoder

        z = self.encoder(inp)

        x = z
        x = layers.Dense(7*7*32)(x)
        x = layers.Reshape((7, 7, 32))(x)
        x = layers.LeakyReLU()(x)
        x = layers.Conv2DTranspose(16, 5, strides=(2, 2), padding='same')(x)
        x = layers.LeakyReLU()(x)
        x = layers.Conv2DTranspose(1, 5, strides=(2, 2), padding='same')(x)

        self.decoder = tf.keras.Model(z, x)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(None, 28, 28, 1), dtype=tf.float32),
        ]
    )
    @u.tf_scope
    def encode(self, img):
        return self.encoder(img)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(None, 2), dtype=tf.float32),
        ]
    )
    @u.tf_scope
    def decode(self, z):
        return self.decoder(z)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ]
    )
    @u.tf_scope
    def sample_latents(self, n):
        """Generate a batch of latent vectors."""
        return tf.random.normal((n, self.latent_dim))

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ]
    )
    @u.tf_scope
    def sample(self, n):
        return self.decode(tf.random.normal((n, 2)))

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(None, 28, 28, 1), dtype=tf.float32),
            (
                tf.TensorSpec(shape=(None, 28, 28, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 2), dtype=tf.float32),
            ),
        ]
    )
    @u.tf_scope
    def loss_fn(self, tar, outs):
        """Loss function for the BIR-AAE model."""

        reconstructed, z = outs

        # reconstruction loss
        rec_loss = tf.reduce_mean(tf.square(tar - reconstructed))

        return rec_loss


    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(None, 28, 28, 1), dtype=tf.float32),
        ]
    )
    @u.tf_scope
    def call(self, inp):
        """
        Encode and decode

        >>> m = BirAae()
        >>> r, z = m(tf.ones((2, 28, 28, 1)))
        >>> r.shape
        TensorShape([2, 28, 28, 1])
        >>> z.shape
        TensorShape([2, 2])
        """
        z = self.encoder(inp)
        reconstructed = self.decoder(z)
        return reconstructed, z


    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ],
    )
    @u.tf_scope
    def visualize_latents(self, n):
        """
        Visualize the latent space.

        >>> m = BirAae()
        >>> imgs = m.visualize_latents(10)
        >>> imgs.shape
        TensorShape([280, 280, 1])
        """

        # grid of latents
        latents_batch = tf.meshgrid(
            tf.linspace(-2., 2., n),
            tf.linspace(-2., 2., n),
        )
        latents_batch = tf.stack(latents_batch, axis=-1)
        latents_batch = tf.reshape(latents_batch, (n*n, 2))
        # decode latents
        imgs = self.decode(latents_batch)
        imgs = (imgs + 1.) / 2. * 255.
        imgs = tf.clip_by_value(imgs, 0, 255)
        imgs = tf.cast(imgs, tf.uint8)
        # reshape to grid
        imgs = tf.reshape(imgs, (n, n, 28, 28, 1))
        imgs = tf.transpose(imgs, (0, 2, 1, 3, 4))
        imgs = tf.reshape(imgs, (n*28, n*28, 1))

        return imgs



if __name__ == '__main__':

    # construct and save model to 'models/ae'

    model = AE(
        latent_dim=2,
    )

    model.build((None, 28, 28, 1))

    model.save('models/ae')
