import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os, random
import cv2
from glob import glob
import sklearn
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator, load_img
from keras._tf_keras.keras.utils import to_categorical
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Activation, Dropout, Dense, Flatten, BatchNormalization, Conv2D, MaxPooling2D
from keras._tf_keras.keras.optimizers import RMSprop
from keras._tf_keras.keras import backend as K
from keras._tf_keras.keras.preprocessing import image
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path
from PIL import Image
def preprocess_images(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray, (5,5), 1)
	thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
	return thresh
def find_largest_contour(thresh):
	contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	largest_contour = max(contours, key=cv2.contourArea)
	return largest_contour
def extract_sudoku_grid(img, contour):
	epsilon = 0.02 * cv2.arcLength(contour, True)
	approx = cv2.approxPolyDP(contour, epsilon, True)

	if len(approx) == 4:
		pts = approx.reshape(4,2)
		rect = np.zeros((4,2), dtype="float32")

		s = pts.sum(axis=1)
		rect[0] = pts[np.argmin(s)]
		rect[2] = pts[np.argmin(s)]

		diff = np.diff(pts, axis=1)
		rect[1] = pts[np.argmin(diff)]
		rect[3] = pts[np.argmin(diff)]

		(tl, tr, br, bl) = rect
		widthA = np.linalg.norm(br - bl)
		widthB = np.linalg.norm(tr - tl)
		maxWidth = max(int(widthA), int(widthB))

		
