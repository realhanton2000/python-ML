from lrCostFunction import lrCostFunction
from oneVsAll import oneVsAll
from predictOneVsAll import predictOneVsAll
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def predictX(index):

    script_dir = os.path.dirname(__file__)
    rel_path = 'ex3data1.mat'
    abs_path = os.path.join(script_dir, rel_path)

    mat_contents = sio.loadmat(abs_path)
    X = mat_contents['X']
    y = mat_contents['y']
    X = abs(X)

    #arr = np.reshape(X[2], (20,20))
    #plt.imshow(arr, cmap='gray', vmin=0, vmax=1)
    # plt.show()

    #theta_t = np.matrix('-2.;-1.;1.;2.')

    #X_t = np.ones((5,1))

    #X_t_2 = []
    #for i in range(1,16):
    #  X_t_2.append(float(i))

    #X_t_2 = np.arange(1, 16)
    #X_t_2 = np.reshape(X_t_2, (3, 5)) / 10
    #X_t_2 = np.transpose(X_t_2)
    #X_t = np.concatenate((X_t, X_t_2), axis=1)

    #y_t = np.matrix('1.;0.;1.;0.;1.')

    #lambda_t = 3.;

    #J, grad = lrCostFunction(theta_t, X_t, y_t, lambda_t)

    #print(J)
    #print(grad)

    lambda_t = 0.1
    num_labels = 10

    all_theta = oneVsAll(X, y, num_labels, lambda_t)

    print(all_theta)

    p = predictOneVsAll(all_theta, X)

    p = np.reshape(p, (p.size,1))

    print(np.average(p == y) * 100)

    #path = 'D:\\dev\\machine-learning\\machine-learning-ex3\\ex3\\test.bmp'
    #oneimage = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    #oneimage = oneimage / 255
    
    print(index)

    oneimage = X[index]

    oneimage = np.reshape(oneimage, (1, oneimage.size))
    r = predictOneVsAll(all_theta, oneimage)

    return r

    #print(r)

    #arr = np.reshape(oneimage, (20,20))
    #plt.imshow(arr, cmap='gray', vmin=0, vmax=1)
    #plt.show()

print(predictX(2333))

