import pickle
import numpy as np
import cv2 as cv
from skimage import color
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import cv2

def rgb_clahe_justl(in_rgb_img): 
    bgr = in_rgb_img[:,:,[2,1,0]] # flip r and b
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    return lab[:,:,0]

def lab_to_lab2(lab):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(lab)
    
X_train = open("/content/gdrive/My Drive/Nucleus Data/training_data_256.pickle","rb")
Y_train = open("/content/gdrive/My Drive/Nucleus Data/training_label_256.pickle","rb")

X_test = open("/content/gdrive/My Drive/Nucleus Data/testing_data_256.pickle","rb")
Y_test = open("/content/gdrive/My Drive/Nucleus Data/testing_label_256.pickle","rb")

X_train = pickle.load(X_train)
Y_train = pickle.load(Y_train)
X_test = pickle.load(X_test)
Y_test = pickle.load(Y_test)



X_test_mod = []
for i in range(len(X_test)):
  X_test_mod.append(rgb_clahe_justl(X_test[i]))

X_test_mod_2 = []
for i in range(len(X_test_mod)):
  X_test_mod_2.append(lab_to_lab2(X_test_mod[i]))

X_test_mod_2 = np.stack((X_test_mod_2,)*3, axis=-1)


X_test_mod_1 = []
for i in range(len(X_test_mod_2)): 
  if X_test_mod_2[i].mean()>127:
    X_test_mod_1.append(255 - X_test_mod_2[i])
  else:
    X_test_mod_1.append(X_test_mod_2[i])
X_test_mod = np.array(X_test_mod)
X_test_mod_1 = np.array(X_test_mod_1)

