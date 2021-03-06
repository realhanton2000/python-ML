import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

script_dir = os.path.dirname(__file__)
rel_path = 'test.bmp'
abs_path = os.path.join(script_dir, rel_path)

oneimage = cv2.imread(abs_path, cv2.IMREAD_GRAYSCALE)
oneimage = oneimage / 255
oneimage = np.reshape(oneimage, (1, oneimage.size))

print(oneimage.shape)