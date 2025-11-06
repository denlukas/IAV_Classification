import tensorflow as tf
import keras


@keras.saving.register_keras_serializable()
class MatthewsCorrelationCoefficient(keras.metrics.Metric):
    def __init__(self, name='matthews_correlation_coefficient', **kwargs):
        super().__init__(name=name, **kwargs)
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.tn = self.add_weight(name='tn', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Flatten and cast
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]) >= 0.5, tf.int32)

        # Confusion matrix values
        tp = tf.reduce_sum(tf.cast((y_pred == 1) & (y_true == 1), self.dtype))
        tn = tf.reduce_sum(tf.cast((y_pred == 0) & (y_true == 0), self.dtype))
        fp = tf.reduce_sum(tf.cast((y_pred == 1) & (y_true == 0), self.dtype))
        fn = tf.reduce_sum(tf.cast((y_pred == 0) & (y_true == 1), self.dtype))

        # Accumulate
        self.tp.assign_add(tp)
        self.tn.assign_add(tn)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)

    def result(self):
        numerator = self.tp * self.tn - self.fp * self.fn
        denominator = tf.math.sqrt(
            (self.tp + self.fp) *
            (self.tp + self.fn) *
            (self.tn + self.fp) *
            (self.tn + self.fn)
        )
        return tf.math.divide_no_nan(numerator, denominator)

    def reset_states(self):
        for var in [self.tp, self.tn, self.fp, self.fn]:
            var.assign(0.0)
