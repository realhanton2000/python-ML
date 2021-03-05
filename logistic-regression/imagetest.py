import matplotlib.pyplot as plt
import cv2
import numpy as np

path = 'D:\\dev\\python-ML\\logistic-regression\\test.bmp'

oneimage = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
oneimage = oneimage / 255
oneimage = np.reshape(oneimage, (1, oneimage.size))

print(oneimage.shape)