from keras.models import model_from_json
from keras.optimizers import RMSprop, Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras.backend import set_image_dim_ordering

import math
import numpy as np


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
 

# learning rate schedule callback
def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.00001
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate

lrate = LearningRateScheduler(step_decay)

# checkpoint callback
checkpoint = ModelCheckpoint("weights/tiramisu_weights.best.hdf5", monitor='val_acc', verbose=2,
                             save_best_only=True, save_weights_only=False, mode='max')

# early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=0, mode='auto')

callbacks_list = [checkpoint, lrate, early_stopping]


optimizer = RMSprop(lr=0.001, decay=0.0000001)
# optimizer = SGD(lr=0.01)
# optimizer = Adam(lr=1e-3, decay=0.995)

tiramisu.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

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
