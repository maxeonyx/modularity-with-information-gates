"""
Bounded information-rate adversarial autoencoder (BIR-AAE) for MNIST. This
demonstrates the use of an MMD-GAN loss on an autoencoder bottleneck to enforce
normally distributed latent variables. Noise can be added to the latent space to
limit the information rate of the bottleneck.

This is a tensorflow implementation of the BIR-VAE model described in the paper
"Bounded Information-Rate Variational Autoencoders" by Braithwaite et al. (2018).
The model is trained to reconstruct images in the MNIST dataset.
"""

from einml.prelude import *

class BirAae(Model):

    @u.tf_scope
    def __init__(
        self,
        latent_dim = 2,
        mmd_kernel_r = 1.0,
        mmd_weight = 100.0,
        name=None
    ):
        super().__init__(name=name)

        self.latent_dim = latent_dim
        self.mmd_kernel_r = mmd_kernel_r
        self.mmd_weight = mmd_weight

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

        # mmd loss
        mmd_loss = self.mmd(z, tf.random.normal(tf.shape(z)))

        return rec_loss + self.mmd_weight * mmd_loss


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
            tf.TensorSpec(shape=(None, 2), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 2), dtype=tf.float32),
        ]
    )
    @u.tf_scope
    def kernel(self, a, b):
        """
        Gaussian radial basis function.
        Takes two vectors a and b and returns |a| X |b| matrix of kernel values.

        >>> m = BirAae()
        >>> k = m.kernel(tf.ones((2, 2)), tf.ones((2, 2)))
        >>> k.shape
        TensorShape([2, 2])
        >>> k.numpy()
        array([[1.        , 0.36787945],
        """
        a = tf.expand_dims(a, 0)
        b = tf.expand_dims(b, 1)
        return tf.exp(-tf.reduce_sum(tf.math.squared_difference(a, b), axis=-1) / self.mmd_kernel_r)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(None, 2), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 2), dtype=tf.float32),
        ]
    )
    @u.tf_scope
    def mmd(self, y_p, y_t):
        """
        Maximum Mean Discrepancy - loss function between two empirical distributions.
        Takes two datasets x and y and returns a value
        between 0 and 2. (actually, it can go slightly less than 0)

        >>> m = BirAae()
        >>> d = m.mmd(tf.zeros([10, 2]), tf.zeros([10, 2]))
        >>> d.shape
        TensorShape([])
        >>> d.numpy()
        0.0
        """
        m = tf.cast(tf.shape(y_p)[0], tf.float32)
        n = tf.cast(tf.shape(y_t)[0], tf.float32)

        sum_1 = tf.reduce_sum(self.kernel(y_p, y_p))
        term_1 = (sum_1)/(m*m)

        sum_2 = tf.reduce_sum(self.kernel(y_t, y_t))
        term_2 = sum_2/(n*n)

        sum_3 = tf.reduce_sum(self.kernel(y_p, y_t))
        term_3 = sum_3/(m*n)

        return term_1 + term_2 - 2*term_3

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

    # construct and save model to 'models/bir_aae'

    model = BirAae(
        latent_dim=2,
        mmd_kernel_r=1,
        mmd_weight=100,
    )

    model.build((None, 28, 28, 1))

    model.save('models/bir_aae')
