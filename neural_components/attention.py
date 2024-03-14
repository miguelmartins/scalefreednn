import tensorflow as tf

from neural_components.custom_ops import sample_min_max_scaling, standardized_general_cosine_similarity
from neural_components.fractal_geometry import LocalSingularityStrength, OrdinaryLeastSquares


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


class SpatialSingularityRecalibration(tf.keras.layers.Layer):
    def __init__(self, r, max_scale, **kwargs):
        super(SpatialSingularityRecalibration, self).__init__(**kwargs)
        self.max_scale = max_scale
        self.r = r
        self.scales = [2 ** i for i in range(1, max_scale + 1)]
        self.bn = tf.keras.layers.BatchNormalization()

    def build(self, input_shape):
        self.conv_list = []
        for r in self.scales:
            self.conv_list.append(tf.keras.layers.DepthwiseConv2D(kernel_size=(r, r),
                                                                  depth_multiplier=1,
                                                                  trainable=False,
                                                                  activation=None,
                                                                  padding="SAME",
                                                                  depthwise_initializer=tf.keras.initializers.Ones()))
        self.scales = tf.cast(self.scales, dtype=tf.float32)

        num_channels = input_shape[-1]
        self.gap = tf.keras.layers.GlobalAvgPool2D()
        self.w1 = tf.keras.layers.Dense(num_channels // self.r,
                                        activation='relu')
        self.w2 = tf.keras.layers.Dense(num_channels,
                                        activation='sigmoid')
        super(SpatialSingularityRecalibration, self).build(input_shape)

    def call(self, x, training=False):
        # TODO: debug sample_standard
        # x = sample_standard(x)  # this step can be removed if one ensures that x is non-negative
        x = sample_min_max_scaling(x)
        measures = tf.stack([conv(x) for conv in self.conv_list], axis=-1)
        alphas = OrdinaryLeastSquares(max_scale=self.max_scale)(measures)
        squeeze = self.gap(alphas)
        excite = self.w2(self.w1(squeeze))
        log_measures = tf.math.log(measures + tf.keras.backend.epsilon())
        log_r = tf.math.log(self.scales)
        ols_estimate = tf.tensordot(alphas, log_r, axes=0)
        sim = standardized_general_cosine_similarity(log_measures, ols_estimate)
        alpha_norm = self.bn(alphas, training=training)
        x = x * excite[:, tf.newaxis, tf.newaxis, :]
        return (sim * tf.nn.sigmoid(alpha_norm)) + ((1 - sim) * x)


class WeightedSpatialSingularityRecalibration(tf.keras.layers.Layer):
    def __init__(self, r, max_scale, **kwargs):
        super(WeightedSpatialSingularityRecalibration, self).__init__(**kwargs)
        self.max_scale = max_scale
        self.r = r
        self.scales = [2 ** i for i in range(1, max_scale + 1)]
        self.bn = tf.keras.layers.BatchNormalization()

    def build(self, input_shape):
        self.conv_list = []
        for r in self.scales:
            self.conv_list.append(tf.keras.layers.DepthwiseConv2D(kernel_size=(r, r),
                                                                  depth_multiplier=1,
                                                                  trainable=False,
                                                                  activation=None,
                                                                  padding="SAME",
                                                                  depthwise_initializer=tf.keras.initializers.Ones()))
        self.scales = tf.cast(self.scales, dtype=tf.float32)
        self.k = self.add_weight(name='kappa',
                                 shape=[len(self.scales)],
                                 initializer=tf.keras.initializers.GlorotUniform(),
                                 dtype=tf.float32,
                                 trainable=True)
        num_channels = input_shape[-1]
        self.gap = tf.keras.layers.GlobalAvgPool2D()
        self.w1 = tf.keras.layers.Dense(num_channels // self.r,
                                        activation='relu')
        self.w2 = tf.keras.layers.Dense(num_channels,
                                        activation='sigmoid')
        super(WeightedSpatialSingularityRecalibration, self).build(input_shape)

    def call(self, x, training=False):
        # TODO: debug sample_standard
        # x = sample_standard(x)  # this step can be removed if one ensures that x is non-negative
        x = sample_min_max_scaling(x)
        measures = tf.stack([conv(x) for conv in self.conv_list], axis=-1)
        alphas = OrdinaryLeastSquares(max_scale=self.max_scale)(measures)
        squeeze = self.gap(alphas)
        excite = self.w2(self.w1(squeeze))
        log_measures = tf.math.log(measures + tf.keras.backend.epsilon())
        log_r = tf.math.log(self.scales)
        k_ = tf.nn.sigmoid(self.k)
        log_k = tf.nn.sigmoid(tf.math.log(k_ + tf.keras.backend.epsilon()))
        #weighted_ols_estimate = tf.tensordot(alphas, log_k, axes=0) - log_k + tf.tensordot(alphas, log_r, axes=0)
        weighted_ols_estimate = tf.tensordot(alphas, log_k, axes=0) + tf.tensordot(alphas, log_r, axes=0)
        #weighted_ols_estimate = log_k + tf.tensordot(alphas, log_r, axes=0)
        sim = standardized_general_cosine_similarity(log_measures, weighted_ols_estimate)

        alpha_norm = self.bn(alphas, training=training)
        x = x * excite[:, tf.newaxis, tf.newaxis, :]
        return (sim * tf.nn.sigmoid(alpha_norm)) + ((1 - sim) * x)


class SSRSpatial(tf.keras.layers.Layer):
    def __init__(self, r, max_scale, **kwargs):
        super(SSRSpatial, self).__init__(**kwargs)
        self.alpha_layer = LocalSingularityStrength(max_scale=max_scale)
        self.r = r

    def build(self, input_shape):
        num_channels = input_shape[-1]
        self.gap = tf.keras.layers.GlobalAvgPool2D()
        self.w1 = tf.keras.layers.Dense(num_channels // self.r,
                                        activation='relu')
        self.w2 = tf.keras.layers.Dense(num_channels,
                                        activation='sigmoid')
        self.beta = self.add_weight(name='beta',
                                    shape=(1),
                                    initializer=tf.keras.initializers.Zeros(),
                                    dtype=tf.float32,
                                    trainable=True)
        super(SSRSpatial, self).build(input_shape)

    def call(self, x):
        alphas = self.alpha_layer(x)
        squeeze = self.gap(alphas)
        excite = self.w2(self.w1(squeeze))
        return (x * excite[:, tf.newaxis, tf.newaxis, :]) + (self.beta * (tf.nn.sigmoid(alphas)))


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
