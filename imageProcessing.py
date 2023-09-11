from cv2 import cv2
import numpy as np
from skimage.color import rgb2gray, rgba2rgb
from skimage.filters import threshold_otsu
from skimage.util import invert
from PIL import Image
import utils
import matplotlib.pyplot as plt

def preProcessing(path):
    img = cv2.imread(path)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # imgThresh = cv2.adaptiveThreshold(imgGray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    retval, imgOtsuInv = cv2.threshold(imgGray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV + cv2.ADAPTIVE_THRESH_MEAN_C)
    retval, imgOtsu = cv2.threshold(imgGray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY + cv2.ADAPTIVE_THRESH_MEAN_C)
    contours, heirarchy = cv2.findContours(imgOtsuInv, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    mx = (0,0,0,0)
    mx_area = 0
    for cont in contours:
        x,y,w,h = cv2.boundingRect(cont)
        area = w*h
        if area > mx_area:
            mx = x,y,w,h
            mx_area = area
    x,y,w,h = mx
    a = x + w//2
    b = y + h//2
    if (h >= w):
        x = a - h//2
        y = b - h//2
        characterCropped=imgOtsuInv[y-20: y+h+20, x-20:x+h+20]
    else:
        x = a - w//2
        y = b - w//2
        characterCropped=imgOtsuInv[y-20: y+w+20, x-20:x+w+20]
    imgShape = characterCropped.shape
    if (imgShape[-1] == 4):
        imgGray2 = rgb2gray(rgba2rgb(characterCropped))
    else:
        imgGray2 = rgb2gray(characterCropped)
    imgOtsu2 = imgGray2 > threshold_otsu(imgGray2)
    np.unique(imgOtsu2)
    imgTemp = Image.fromarray(np.float64(imgOtsu2))
    imgTemp1 = imgTemp.resize((28, 28))
    imgTemp2 = np.asarray(imgTemp1)
    imgResult = imgTemp2.astype('float64')
    print(imgResult.dtype, imgResult.shape, type(imgResult))
    plt.imshow(imgResult, cmap = plt.cm.binary)   
    plt.show()
    return imgResult