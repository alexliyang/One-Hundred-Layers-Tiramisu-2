from keras.models import model_from_json
from cv2 import imread, imwrite
import numpy as np
from helper import normalized

with open('tiramisu_fc_dense_model.json') as model_file:
    tiramisu = model_from_json(model_file.read())
    tiramisu.load_weights('weights/tiramisu_weights.best.hdf5')
    img = np.dstack((normalized(imread('CamVid/train/0001TP_006690.png')), imread('CamVid/train/superpixel/0001TP_006690.png')))[:224, :224]
    img = tiramisu.predict(np.array([img]), batch_size=1)
    img = np.argmax(img[0], axis=1).reshape(224, 224, 1)
    print(img)
    imwrite('predict.png', img)
