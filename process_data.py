import os
import numpy as np
# import pandas
# import matplotlib
# import matplotlib.image as mpimg
import imutils
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import keras
from keras.utils import to_categorical
import cv2


def one_hot_encoded(y):  # representation of categories as binary vectors
    p = list(np.unique(y))
    dictionary = dict()
    final_result = []
    file = open('classes.txt', 'w')
    for i in range(len(p)):
        dictionary[p[i]] = i  # Creates a dictionary where p[i] is the category and i is the index
        file.write(str(i) + ' : ' + str(p[i]) + '\n')
    file.close()

    for i in y:
        actual = np.zeros(len(p), dtype=int)  # Fills an array of zeros
        actual[dictionary[i[0]]] = 1  # Sets the actual class to 1
        final_result.append(actual)  # Appends to the final result to be returned
    return np.array(final_result)

script_dir = os.path.dirname(__file__)
dataset_dir = 'dataset'
path = os.path.join(script_dir, dataset_dir)
folders = os.listdir(path)  # Different subjects
folders = set(folders)

different_classes = os.listdir(path+'/'+'00/train_pose')
different_classes = set(different_classes)

print("The different classes that exist in this dataset are:")
classes = dict()
names_of_classes = dict()
for i in different_classes:
    classes[int(i.split('_')[0])] = '_'.join(i.split('_')[1:])
    names_of_classes['_'.join(i.split('_')[1:])] = int(i.split('_')[0])
print(classes)

# PREP TRAINING DATA

x_data = []
y_data = []
y_name = []

threshold = 200

for i in folders:
    subject = path + '/' + i + '/train_pose'  # Gestures path
    subdir = os.listdir(subject)
    subdir = set(subdir)
    for j in subdir:
        images = os.listdir(subject + '/' + j)  # Images for current gesture
        for k in images:
            img = cv2.imread(subject + '/' + j + '/' + k, cv2.IMREAD_GRAYSCALE)  # Reads image in grayscale
            img = cv2.resize(img, (int(192), int(128)))  # Resize, keeping a similar ratio
            thresholded = cv2.GaussianBlur(img, (5, 5), 0)
            ret, thresholded = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Thresholds image to get black and white with Otsu thresholding
            thresholded = np.asarray(thresholded, dtype=np.float64)  # turns image into float64 array
            x_data.append(thresholded)
            plt.imshow(img, cmap=cm.gray)
            plt.show()
            plt.imshow(thresholded, cmap=cm.gray)
            plt.show()
            y_data.append(int(j.split('_')[0]))  # numeric classes array
            y_name.append(j.split('_')[1])  # classes array
            break
        break
    break
print(len(x_data))

