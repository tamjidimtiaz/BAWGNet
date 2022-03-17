
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
from PIL import Image
import pickle
import numpy as np
import cv2 as cv

def rgb_clahe_justl(in_rgb_img): 
    bgr = in_rgb_img[:,:,[2,1,0]] # flip r and b
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    return lab[:,:,0]

def lab_to_lab2(lab):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(lab)
    
def flip_image_1(image):
    return tf.image.flip_left_right(image)
    

def rotate_image_1(image):
    return tf.image.rot90(image, k=1)

def shear_transform_example(filename,shear_lambda):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    img = transformImg(image_decoded, [[1.0,0,0],[shear_lambda,1.0,0],[0,0,1.0]])
    return img
