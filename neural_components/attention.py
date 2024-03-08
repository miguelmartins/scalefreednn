import tensorflow as tf

from neural_components.fractal_geometry import LocalSingularityStrength


class SingularityStrengthRecalibration(tf.keras.layers.Layer):
    def __init__(self, r, max_scale, **kwargs):
        super(SingularityStrengthRecalibration, self).__init__(**kwargs)
        self.alpha_layer = LocalSingularityStrength(max_scale=max_scale)
        self.r = r

    def build(self, input_shape):
        num_channels = input_shape[-1]
        self.gap = tf.keras.layers.GlobalAvgPool2D()
        self.w1 = tf.keras.layers.Dense(num_channels // self.r,
                                        activation='relu')
        self.w2 = tf.keras.layers.Dense(num_channels,
                                        activation='sigmoid')
        super(SingularityStrengthRecalibration, self).build(input_shape)

    def call(self, x):
        alphas = self.alpha_layer(x)
        squeeze = self.gap(alphas)
        excite = self.w2(self.w1(squeeze))
        return x * excite[:, tf.newaxis, tf.newaxis, :]


class SqueezeExcite(tf.keras.layers.Layer):
    def __init__(self, r, **kwargs):
        super(SqueezeExcite, self).__init__(**kwargs)
        self.r = r

    def build(self, input_shape):
        num_channels = input_shape[-1]
        self.gap = tf.keras.layers.GlobalAvgPool2D()
        self.w1 = tf.keras.layers.Dense(num_channels // self.r,
                                        activation='relu')
        self.w2 = tf.keras.layers.Dense(num_channels,
                                        activation='sigmoid')
        super(SqueezeExcite, self).build(input_shape)

    def call(self, x):
        squeeze = self.gap(x)
        excite = self.w2(self.w1(squeeze))
        return x * excite[:, tf.newaxis, tf.newaxis, :]

    def get_config(self):
        config = super(SqueezeExcite, self).get_config()
        config.update({
            "r": self.r
        })
        return config
