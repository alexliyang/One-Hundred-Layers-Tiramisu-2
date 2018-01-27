from keras.models import model_from_json
from keras.optimizers import RMSprop, Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, Callback
from keras.backend import set_image_dim_ordering

import math
import numpy as np
import tensorflow as tf
import keras.backend as K

set_image_dim_ordering('tf')
np.random.seed(7) # 0bserver07 for reproducibility


class_weighting = [
  0.2595,
  0.1826,
  4.5640,
  0.1417,
  0.5051,
  0.3826,
  9.6446,
  1.8418,
  6.6823,
  6.2478,
  3.0,
  7.3614
]


# load the data
train_data = np.load('./data/train_data.npy')
train_data = train_data.reshape((367, 224, 224, 3))
train_label = np.load('./data/train_label.npy') # [:,:,:-1]

test_data = np.load('./data/test_data.npy')
test_data = test_data.reshape((233, 224, 224, 3))
test_label = np.load('./data/test_label.npy') # [:,:,:-1]


# load the model:
with open('tiramisu_fc_dense_model.json') as model_file:
    tiramisu = model_from_json(model_file.read())

# tiramisu.load_weights("weights/tiramisu_weights.best.hdf5")

class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.
    For more detail, please see paper.

    # Example
    clr = CyclicLR(base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular')
    model.fit(X_train, Y_train, callbacks=[clr])

    Class also supports custom scaling functions:
    clr_fn = lambda x: 0.5 * (1 + np.sin(x * np.pi / 2.))
    clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                   step_size=2000., scale_fn=clr_fn,
                   scale_mode='cycle')
    model.fit(X_train, Y_train, callbacks=[clr])

    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None, new_step_size=None):
        # Resets cycle iterations.
        # Optional boundary/step size adjustment.
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(self.clr_iterations)

    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())

    def on_batch_end(self, epoch, logs=None):
        logs = logs or {}

        self.trn_iterations += 1
        self.clr_iterations += 1
        K.set_value(self.model.optimizer.lr, self.clr())

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
 

# learning rate schedule callback
def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.00001
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate

# lrate = LearningRateScheduler(step_decay)
lrate = CyclicLR(step_size=200)

# checkpoint callback
checkpoint = ModelCheckpoint("weights/tiramisu_weights.best.hdf5", monitor='val_acc', verbose=2,
                             save_best_only=True, save_weights_only=False, mode='max')

# early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=0, mode='auto')

callbacks_list = [lrate, early_stopping]


optimizer = RMSprop(lr=0.001, decay=0.0000001)
# optimizer = SGD(lr=0.01)
# optimizer = Adam(lr=1e-3, decay=0.995)

def mean_iou(y_true, y_pred):
    score, up_opt = tf.metrics.mean_iou(y_true, y_pred, 12)
    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([up_opt]):
        score = tf.identity(score)
    return score

tiramisu.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy", mean_iou])

# Fit the model
history = tiramisu.fit(x=train_data, y=train_label,
                       batch_size=2, epochs=150,
                       callbacks=callbacks_list, 
                       class_weight=class_weighting, 
                       verbose=1, shuffle=True,
                       validation_data=(test_data, test_label))

# This save the trained model weights to this file with number of epochs
tiramisu.save_weights('weights/tiramisu_weights{}.hdf5'.format(nb_epoch))

import matplotlib.pyplot as plt
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
