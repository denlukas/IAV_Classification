import keras

from keras.layers import (InputLayer, Input, Layer, Conv1D, Dense, Dropout, GRU, ReLU,
                          BatchNormalization, GlobalAveragePooling1D)

from Püntener.mcc import MatthewsCorrelationCoefficient

import numpy as np

def make_model(input_shape: tuple[int], mcd:float=0.6) -> keras.Model:
    """
    This function defines a model to process 1D sequential data -- our GFP intensity
    time series. We use dropout as a regularization technique, and Monte Carlo dropout
    specifically for uncertainty quantification.
    """
    inputs = Input(shape=input_shape)
    # Feel free to change anything between here and `model = keras.Model(inputs, outputs)`!

    # We want to extract local patterns from sequential data, therefore we use convolutions.
    # This layer scans the sequence with 64 pattern detectors (=filters) of width 3, moving
    # 2 steps at a time. ReLU keeps only positive activations to add non-linearity.
    x = Conv1D(64, kernel_size=3, strides=2, activation='relu')(inputs)

    # This layer normalizes activations to have stable mean and variance. It helps speed up
    # training and reduce overfitting.
    x = BatchNormalization()(x)

    # Dropout serves as regularization technique, to avoid overfitting during training.
    # The 'MC' stands for Monte Carlo, meaning we can also use dropout during inference.
    # This is not totally usual, but *variational dropout inference* as it's called can
    # be used as an uncertainty estimate. The 'rate' parameter of the dropout indicates
    # what proportion of the connections between the layer above (BatchNormalization)
    # and the one below (another Conv1D) the model 'forgets'.
    x = MCDropout(mcd)(x)

    # These two, we already know from above.
    x = Conv1D(8, kernel_size=3, strides=2, activation='relu')(x)
    #x = BatchNormalization()(x)
    #x = MCDropout(.4)(x)

    # add third layer
    #x = Conv1D(64, kernel_size=3, strides=2, activation='relu')(x)
    #x = BatchNormalization()(x)
    #x = MCDropout(.4)(x)

    # A GRU can capture temporal dependencies. We hope that the Conv1Ds learns to detect
    # the GFP peaks, and the GRU learns if the correct time interval passed between them.
    #x = GRUWithMCDropout(32, return_sequences=True, dropout=.4)(x)
    x = GRUWithMCDropout(16, dropout=.4)(x)

    # This is a fully-connected ('Dense') output layer with one neuron and
    # sigmoid activation, suitable for a binary classification problem like
    # ours and meant to approximate a probability distribution.
    outputs = Dense(1, activation='sigmoid')(x)

    # Now wrap this into keras model:
    model = keras.Model(inputs, outputs)

    # Neural networks learn by gradually adjusting their parameters to reduce prediction error.
    # For each training batch, they compute how the weights should change to improve accuracy,
    # but only apply a small fraction of that change to avoid instability. The idea is to make
    # small, iterative improvements across all datapoints. Early in training, larger updates
    # help the model learn quickly; later, smaller updates are better it. This step size is
    # controlled by the learning rate (=lr), which can be managed by a schedule like
    # ExponentialDecay. Alternatively, you can use callbacks like ReduceLROnPlateau -- but
    # this would be applied elsewhere, have a look!
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-2,
        decay_steps=400,
        decay_rate=0.6)

    # This is where we define how the model learns, what it is trying to optimize, and how
    # we measure if it's doing any good: the metrics we want to track during its training.
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),  # optionally attach lr_schedule
        # the BCE is a good loss function do measure the difference between
        # predicted probabilities / logits and actually true, binary labels.
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            # which stats do we want to track?
            keras.metrics.BinaryAccuracy(name='acc'),
            #  MCC
            MatthewsCorrelationCoefficient(name='mcc'),
            #  area under the ROC curve
            keras.metrics.AUC(name='auc', curve='ROC'),
            #  area under the PR curve
            keras.metrics.AUC(name='auc_pr', curve='PR'),
            #  a metric corresponding to either of the two following questions, both of which are
            #  about the predictions with the highest values:
            #  - Out of the x predictions with the highest value, what share is correct?
            #  - When we sort the predictions in descending order, what share of the actual positives
            #    do we find before we find a negative? Think about the more general case of this!
            keras.metrics.PrecisionAtRecall(0.8, name='precision_at_recall_0.8')
        ],
    )
    return model


# You shouldn't need to adapt these two classes:
@keras.saving.register_keras_serializable(package='Custom', name='MCDropout')
class MCDropout(Dropout):
    def call(self, inputs, training=None, mc_dropout=False, **kwargs):
        return super().call(inputs, training=training or mc_dropout)


@keras.saving.register_keras_serializable(package='Custom', name='GRUWithMCDropout')
class GRUWithMCDropout(Layer):
    def __init__(self, units, return_sequences=False, **kwargs):
        super().__init__()
        self.gru = GRU(units, return_sequences=return_sequences, **kwargs)

    def call(self, inputs, training=None, mc_dropout=False):
        return self.gru(inputs, training=training or mc_dropout)

    def build(self, input_shape):
        self.gru.build(input_shape)
        super().build(input_shape)


# ----------------------------
# Monte Carlo Prediction
# ----------------------------
def monte_carlo_predict_samples(model, x, n_mc_samples=100):
    """
    Performs multiple stochastic forward passes with MC dropout.
    Returns both the mean prediction and all predictions.

    Args:
        model (keras.Model): Your trained model.
        x (np.ndarray or tf.Tensor): Input data.
        n_mc_samples (int): Number of stochastic forward passes.

    Returns:
        y_mean (np.ndarray): Mean prediction over MC samples.
        y_all (np.ndarray): All individual MC predictions.
    """
    preds = []
    for _ in range(n_mc_samples):
        # Enable dropout during inference
        y_pred = model(x, training=True).numpy()
        preds.append(y_pred)

    preds = np.stack(preds)  # shape: (n_mc_samples, n_samples, 1)
    return preds.mean(axis=0), preds