import tensorflow as tf

from neural_components.convolutional import SharedConv2D, FSpecialGaussianInitializer
from neural_components.custom_ops import sample_min_max_scaling


class OrdinaryLeastSquares(tf.keras.layers.Layer):
    def __init__(self, max_scale, **kwargs):
        super(OrdinaryLeastSquares, self).__init__(**kwargs)
        self.max_scale = max_scale

    def call(self, x):
        scales = 2 ** tf.range(1, self.max_scale + 1,
                               dtype=tf.float32)  # Inquiry if adding EPSILON here makes sense
        log_scales = tf.math.log(scales)
        log_measures = tf.math.log(x + tf.keras.backend.epsilon())
        mean_log_scales = tf.reduce_mean(log_scales)
        mean_log_measures = tf.reduce_mean(log_measures, axis=-1)[..., tf.newaxis]  # make it broadcastable
        numerator = (log_measures - mean_log_measures) * (log_scales - mean_log_scales)
        denominator = (log_scales - mean_log_scales) ** 2
        return tf.reduce_sum(numerator, axis=-1) / tf.reduce_sum(denominator, axis=-1)


class OrdinaryLeastSquares_(tf.keras.layers.Layer):
    def __init__(self, max_scale, **kwargs):
        super(OrdinaryLeastSquares_, self).__init__(**kwargs)
        self.max_scale = max_scale

    def call(self, x):
        scales = tf.range(1, self.max_scale + 1,
                          dtype=tf.float32)  # Inquiry if adding EPSILON here makes sense
        log_scales = tf.math.log(scales)
        log_measures = tf.math.log(x + tf.keras.backend.epsilon())
        mean_log_scales = tf.reduce_mean(log_scales)
        mean_log_measures = tf.reduce_mean(log_measures, axis=-1)[..., tf.newaxis]  # make it broadcastable
        numerator = (log_measures - mean_log_measures) * (log_scales - mean_log_scales)
        denominator = (log_scales - mean_log_scales) ** 2
        return tf.reduce_sum(numerator, axis=-1) / tf.reduce_sum(denominator, axis=-1)

    def get_config(self):
        config = super(OrdinaryLeastSquares_, self).get_config()
        config.update({
            "max_scale": self.max_scale
        })
        return config


class LocalSingularityStrength(tf.keras.layers.Layer):
    def __init__(self, max_scale, **kwargs):
        super(LocalSingularityStrength, self).__init__(**kwargs)
        self.max_scale = max_scale
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
        super(LocalSingularityStrength, self).build(input_shape)

    def call(self, x, training=False):
        x = sample_min_max_scaling(x)  # this step can be removed if one ensures that x is non-negative
        measures = tf.stack([conv(x) for conv in self.conv_list], axis=-1)
        alphas = OrdinaryLeastSquares(max_scale=self.max_scale)(measures)
        return self.bn(alphas, training=training)


class LearnableMeasure(tf.keras.layers.Layer):
    def __init__(self, max_scale, trainable=True, **kwargs):
        super(LearnableMeasure, self).__init__(**kwargs)
        self.max_scale = max_scale
        self.trainable = trainable

    def build(self, input_shape):
        self.op_list = []
        for i in range(1, self.max_scale + 1):
            self.op_list.append(SharedConv2D(1, kernel_size=(i, i),
                                             strides=(1, 1),
                                             padding='SAME',
                                             kernel_initializer=FSpecialGaussianInitializer(sigma=i / 2),
                                             activation=None,
                                             use_bias=False,
                                             trainable=self.trainable))

        super(LearnableMeasure, self).build(input_shape)

    def call(self, x):
        measurements = []
        for i in range(self.max_scale):
            # out = self.op_list[i](x) * ((i + 1) ** 2) + 1 add add relu in the conv definition
            out = self.op_list[i](x) * ((i + 1) ** 2)
            out = tf.nn.relu(out) + 1
            measurements.append(out)

        return tf.stack(measurements, axis=-1)


class LocalSingularityStrengthXu(tf.keras.layers.Layer):
    def __init__(self, local_scale, trainable=True, with_bn=True, **kwargs):
        self.local_scale = local_scale
        self.holder = LearnableMeasure(self.local_scale, trainable)
        self.lsf = OrdinaryLeastSquares_(self.local_scale)
        self.holder_bn = tf.keras.layers.BatchNormalization()
        self.with_bn = with_bn
        super(LocalSingularityStrengthXu, self).__init__(**kwargs)

    def call(self, x, training=False):
        x = sample_min_max_scaling(x)  # this follows the authors' implementation: https://github.com/csfengli/FENet
        x = self.holder(x)
        x = self.lsf(x)
        if self.with_bn:
            x = self.holder_bn(x, training=training)
        return x
