import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPUs available:", gpus)
    tf.config.experimental.set_memory_growth(gpus[0], True)
else:
    print("No GPUs found. Please check your installation.")
