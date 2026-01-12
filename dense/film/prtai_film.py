#!/usr/bin/env python3

import tensorflow as tf
import keras
import time
import datetime
import numpy as np
import math
import sys, os, argparse
import subprocess

parser = argparse.ArgumentParser(
                    prog='prtai_film',
                    description='FiLM network for DIRC particle classification')

parser.add_argument('-s', '--s', help='Log compression fall-off scale factor')
args = parser.parse_args()

s = float(args.s)

tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')

os.environ["ROBCLAS_VERBOSE"] = "0"
os.environ["ROCM_INFO_LEVEL"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

print(f"[INFO] Environment set")

program_start = time.time()

infile = "2M22TO90timing.npz"

f = np.load(infile, mmap_mode='r')
TIMES, ANGLES, LABELS = f["TIMES"], f["ANGLES"], f["LABELS"]
nevents = TIMES.shape[0]
time_dim = TIMES.shape[1]
angle_dim = ANGLES.shape[1]

print(f"[INFO] Data size: {sys.getsizeof(TIMES)//10**6} MB")


# ---------------------------------------------------------------
#
#                       PARAMETERS
#
# ---------------------------------------------------------------
class_names = ['Pi+', 'Proton']
num_classes = len(class_names) # Pions or kaons?


batch_size  = 128 # How many events to feed to NN at a time?
nepochs     = 10 # How many epochs?

trainfrac   = 0.7
valfrac     = 0.15
testfrac    = 0.15

datafrac    = 1  # What fraction of data to use?
# ---------------------------------------------------------------

trainfrac, valfrac, testfrac = (datafrac*trainfrac, datafrac*valfrac, datafrac*testfrac)

trainend    = int(np.floor(nevents * trainfrac))
valend      = int(trainend + np.floor(nevents * valfrac))
testend     = int(valend + np.floor(nevents * testfrac))

print(f"[INFO] Using {testend} out of {nevents} available events")

traintimes  = TIMES[:trainend]
trainangles = ANGLES[:trainend]
trainlabels = LABELS[:trainend]
valtimes    = TIMES[trainend:valend]
valangles   = ANGLES[trainend:valend]
vallabels   = LABELS[trainend:valend]
testtimes   = TIMES[valend:testend]
testangles  = ANGLES[valend:testend]
testlabels  = LABELS[valend:testend]



class ScheduledFiLM(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ScheduledFiLM, self).__init__(**kwargs)
        self.lambda_var = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self.ranp_rate = 0.01
        

    def call(self, inputs):
        x, gamma, beta = inputs
        lam = tf.clip_by_value(self.lambda_var, 0.0, 1.0)
        return (1.0 + lam * gamma) * x + lam * beta



class ScheduledFiLMCallback(keras.callbacks.Callback):
    def __init__(self, film_layer, nepochs):
        super(ScheduledFiLMCallback, self).__init__()
        self.film_layer = film_layer
        self.nepochs = nepochs

    def on_epoch_end(self, epoch, logs=None):
        new_lambda = (2*epoch + self.nepochs) / (epoch + self.nepochs) - 0.5
        self.film_layer.lambda_var.assign(new_lambda)
        print(f'\nUpdated FiLM lambda to {new_lambda:.4f}')



class BatchGenerator(keras.utils.Sequence):
    """
    Converts to dense batches for training during runtime.
    Class keeps ordering and supports shuffling each epoch.

    
    Parameters
    ----------
    *args : ndarray
        Sparse or dense matrices to be placed into batches. The arguments should include label data as the last argument.
    >>> train_gen = BatchGenerator(times, angles, labels, *kwargs)
    batch_size : int
        Number of matrices to batch.
    shuffle : bool
        Whether or not to shuffle data on epoch end.
    """

    def __init__(self, *args, batch_size=256, shuffle=True):
        self.args = args
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n = self.args[0].shape[0]
        self.indices = np.arange(self.n)
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(self.n / float(self.batch_size)))
    
    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_list = [np.asarray(data[batch_idx]) for data in self.args]
        
        inputs = batch_list[:-1]
        labels = batch_list[-1]
        
        return tuple(inputs), labels
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def width_function(x, N_max, x_min):
    return N_max * np.log(1.0 / x) / np.log(1.0 / x_min)

def layer_dims(s, L, N_max, N_min=8):
    x_min = 1.0 / (L + 1)
    xs = ((np.arange(L) + 1) / (L + 1)) ** s

    widths = width_function(xs, N_max, x_min)
    widths = np.maximum(widths.astype(int), N_min)

    return widths

L = 5
N_max = 256

widths = layer_dims(s, L, N_max)
print(widths)

dropout = 0.15

# Time Data Branch
hist_input = keras.Input(shape=(time_dim,))

h = keras.layers.LayerNormalization()(hist_input)
h = keras.layers.Dense(widths[0], activation='gelu')(h)
h = keras.layers.Dense(widths[1], activation='gelu')(h)
h = keras.layers.Dense(widths[2], activation='gelu')(h)

#h = keras.layers.Dropout(dropout)(h)


# Angle Data Branch
@keras.utils.register_keras_serializable(package='Custom', name='ScaleAngles')
class ScaleAngles(keras.layers.Layer):
    def __init__(self, initial_scale=1.0, dtype='bfloat16', **kwargs):
        super().__init__(**kwargs)
        self.initial_scale = initial_scale
        self._dtype = dtype

    def build(self, input_shape):
        # trainable scalar weight tracked by the layer
        self.scale = self.add_weight(
            name='scale',
            shape=(),
            initializer=tf.keras.initializers.Constant(self.initial_scale),
            trainable=True,
            dtype=self._dtype
        )

    def call(self, x):
        # ensure same dtype for multiplication
        s = tf.cast(self.scale, x.dtype)
        angle = x[:, 4:] * s
        rest = x[:, :4]
        return tf.concat([angle, rest], axis=1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'initial_scale': self.initial_scale, 'dtype': self._dtype})
        return cfg

angle_input = keras.Input(shape=(angle_dim,))
scaled_input = ScaleAngles(initial_scale=1.0, dtype='bfloat16')(angle_input)

a = keras.layers.LayerNormalization()(scaled_input)
a = keras.layers.Dense(widths[1], activation='gelu')(a)
a = keras.layers.Dense(widths[2], activation='gelu')(a)

# Produce FiLM parameters
gamma = keras.layers.Dense(widths[2], kernel_regularizer=keras.regularizers.L2(1e-04), name='gamma')(a)
# push gamma towards 1 so that network is encouraged to use FiLM layer, not just ignore it
#gamma = keras.layers.Lambda(lambda g: tf.exp(0.1*g))(gamma_raw)
beta  = keras.layers.Dense(widths[2], activation='linear', name='beta')(a)

# FiLM layer
h_mod = keras.layers.Multiply()([h, gamma])
h_mod = keras.layers.Add()([h_mod, beta])

# Combined layers and output
drop = keras.layers.Dropout(dropout, name='dropout')(h_mod)
x = keras.layers.Dense(widths[3], activation='gelu')(drop)
x = keras.layers.Dense(widths[4], activation='gelu')(x)
out = keras.layers.Dense(num_classes, activation='softmax', name='output')(x)

model = keras.Model(inputs=[hist_input, angle_input], outputs=out)
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)
model.summary()


train_gen   = BatchGenerator(traintimes, trainangles, trainlabels, batch_size=batch_size, shuffle=True)
val_gen     = BatchGenerator(valtimes, valangles, vallabels, batch_size=batch_size, shuffle=True)
test_gen    = BatchGenerator(testtimes, testangles, testlabels, batch_size=batch_size, shuffle=True)


model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=nepochs, 
    #callbacks=[ScheduledFiLMCallback(film, nepochs)],
)

test_loss, test_acc = model.evaluate(
    test_gen, verbose=2
)

print('\nTest accuracy:', test_acc)
print('Test loss:', test_loss)

subprocess.run('mkdir models', shell=True)
date_str = datetime.date.today().isoformat()
i = 1
while True:
    out_name = f"models/{date_str}_model{i}.keras"
    if not os.path.exists(out_name):
        break
    i += 1

model.save(out_name)
print(f"Saved model to {out_name}")

program_end = time.time()
print(f"Done in {program_end - program_start}s")