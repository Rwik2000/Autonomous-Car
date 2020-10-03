## PV, DNE
import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob
import math
import time
import random
from scipy.spatial import distance
import sys

window = None
fps = 30
def chooseWindow(event, x, y, flags, param):
    global window
    if event == cv2.EVENT_LBUTTONDOWN:
        window = cv2.resize(frame[y - 50: y + 50, x - 50:x + 50], (200,200))
        windowTemp = cv2.resize(res[y - 50: y + 50, x - 50:x + 50], (200,200))
        cv2.imshow('small', windowTemp)
        cv2.imshow('small1', window)

def showColour(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(window[y, x])
        col = np.zeros((200,200,3))
        col[True] = window[y, x]
        cv2.imshow('colour', col)

cv2.namedWindow('res')
cv2.setMouseCallback('res', chooseWindow)


cv2.namedWindow('small')
cv2.setMouseCallback('small', showColour)

res = None
frame = None

def executeVideo():
    global res, frame
    path = 'inputs/videos/' + sys.argv[1]
    print(path)
    cap = cv2.VideoCapture(path)
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameSum = frame.astype(float)
    i = 0
    while(cap.isOpened()):
        startTime = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.GaussianBlur(frame,(7,7),0)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # frame = hsv
        lower_blue = np.array([0,0, 0])
        upper_blue = np.array([40,255,255])

        # lower_blue = np.array([0,0, 0])
        # upper_blue = np.array([255,75,150])


        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        lower_blue = np.array([160,0, 0])
        upper_blue = np.array([180,255,255])
        mask += cv2.inRange(hsv, lower_blue, upper_blue)


        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(frame,frame, mask= mask)

        cv2.imshow('frame',frame)
        cv2.imshow('mask',mask)
        cv2.imshow('res',res)
        while time.time() - startTime < 1/fps:
            True
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit(0)
        
    cap.release()
    
    
            
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    executeVideo()
    # executeImage()

    