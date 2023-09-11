from cv2 import cv2
import numpy as np
from skimage.color import rgb2gray, rgba2rgb
from skimage.filters import threshold_otsu
from PIL import Image

def preProcessing(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Image gray", imgGray)
    return imgGray

def getContours(img):
    biggest = np.array([])
    maxArea = 0
    imgCanny = cv2.Canny(img, 200, 200)
    contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    return biggest

def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4,1,2), np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis = 1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew


def getWarp(img, biggest, widthImg, heightImg):
    biggest = reorder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0,0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    imgCropped = imgOutput[10:imgOutput.shape[0]-10, 10:imgOutput.shape[1]-10]
    imgCropped = cv2.resize(imgCropped, (widthImg, heightImg))
    return imgCropped

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def getCharacter(img):
    retval, thresh_gray = cv2.threshold(img, thresh=100, maxval=255, type=cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(thresh_gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
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
        characterCropped=img[y-20: y+h+20, x-20: x+h+20]
    else:
        x = a - w//2
        y = b - w//2
        characterCropped=img[y-20: y+w+20, x-20:x+w+20]
    return characterCropped

def finalResult(img):
    imgShape = img.shape
    if (imgShape[-1] == 4):
        imgGray = rgb2gray(rgba2rgb(img))
    else:
        imgGray = rgb2gray(img)
    # imgInverted = np.invert(imgGray)
    imgOtsu = imgGray > threshold_otsu(imgGray)
    np.unique(imgOtsu)
    # cv2.imshow("img otsu", imgOtsu)
    # cv2.waitKey(0)
    imgTemp = Image.fromarray(np.float64(imgOtsu))
    imgTemp1 = imgTemp.resize((28, 28))
    imgTemp2 = np.asarray(imgTemp1)
    imgResult = imgTemp2.astype('float64')
    return imgResult