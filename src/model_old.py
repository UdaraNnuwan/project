import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

@keras.utils.register_keras_serializable(package="Custom")
class FiLM(layers.Layer):
    """Feature-wise Linear Modulation (FiLM) layer for conditioning feature maps on context."""
    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.units = int(units)
        self.modulation = layers.Dense(2 * self.units)

    def call(self, inputs):
        x, context = inputs
        gamma_beta = self.modulation(context)
        gamma, beta = tf.split(gamma_beta, 2, axis=-1)
        gamma = tf.expand_dims(gamma, axis=1)
        beta  = tf.expand_dims(beta, axis=1)
        return x * (1.0 + gamma) + beta

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units})
        return cfg


def film_block(x, context, units: int, conv_k: int = 3, name: str | None = None):
    """Reusable Conv1D -> FiLM -> ReLU block."""
    x = layers.Conv1D(units, conv_k, padding="same", name=None if name is None else f"{name}_conv")(x)
    x = FiLM(units, name=None if name is None else f"{name}_film")([x, context])
    x = layers.Activation("relu", name=None if name is None else f"{name}_relu")(x)
    return x


def build_film_ae(window_size: int, n_features: int, context_dim: int,
                  units: int = 64, latent: int = 64) -> keras.Model:
    """Context-conditioned autoencoder using FiLM blocks (Option A)."""
    x_in = keras.Input(shape=(window_size, n_features), name="x")
    c_in = keras.Input(shape=(context_dim,), name="context")

    # Encoder
    h = film_block(x_in, c_in, units, name="enc1")
    h = film_block(h,    c_in, units, name="enc2")
    h = layers.GlobalAveragePooling1D(name="gap")(h)
    z = layers.Dense(latent, activation="relu", name="z")(h)

    # Decoder
    d = layers.Dense(window_size * units, activation="relu", name="dec_dense")(z)
    d = layers.Reshape((window_size, units), name="dec_reshape")(d)
    d = film_block(d, c_in, units, name="dec1")
    d = layers.Conv1D(units, 3, padding="same", activation="relu", name="dec_conv")(d)
    out = layers.Conv1D(n_features, 3, padding="same", name="out")(d)

    model = keras.Model([x_in, c_in], out, name="film_ae")
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model
