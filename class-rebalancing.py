from cv2 import imread
import numpy as np
import os

DataPath = './CamVid/'

def count(mode):
    s = np.zeros(12)
    with open(DataPath + mode +'.txt') as f:
        txt = f.readlines()
        txt = [line.split(' ') for line in txt]
    for i in range(len(txt)):
        label = imread(os.getcwd() + txt[i][1][7:][:-1])[:224, :224, 0].flatten()
        unique, counts = np.unique(label, return_counts=True)
        # assert(np.maximum(unique) < 12)
        d = dict(zip(unique, counts))
        for i in d:
            s[i] += d[i]
    return s

s = count('train') + count('val') + count('test')
# s = np.amin(s) / s * 10
s /= np.sum(s)
s = np.median(s) / s
print(s)
