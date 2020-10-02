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

heightStart = 20
heightSkipInterval = 8
nFrames = 1

def getLanes(thresh):
    height = thresh.shape[0]
    width = thresh.shape[1]
    # seedThresh = np.zeros_like(thresh)
    seedThresh = thresh.copy()
    thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    for i in range(0, 14):
        coord = (int(width/2),height - heightSkipInterval*i -heightStart)
        row = thresh[coord[1]]
        isLane = (row != 0)
        leftLaneIndex = None
        rightLaneIndex = None
        for j in range(0, coord[0]):
            if leftLaneIndex != None and rightLaneIndex != None:
                break
            if leftLaneIndex == None and isLane[coord[0] - j] == True:
                leftLaneIndex = (int(width/2) - j,height - heightSkipInterval*i -heightStart)
            if rightLaneIndex == None and isLane[coord[0] + j] == True:
                rightLaneIndex = (int(width/2) + j,height - heightSkipInterval*i -heightStart)
        if leftLaneIndex == None:
            leftLaneIndex = coord
        if rightLaneIndex == None:
            rightLaneIndex = coord
            
        cv2.circle(seedThresh, rightLaneIndex, 3, (0, 0, 255), cv2.FILLED, cv2.LINE_AA)
        cv2.circle(seedThresh, leftLaneIndex, 3, (255, 255, 0), cv2.FILLED, cv2.LINE_AA)
        # print(closest_node(coord, ))
        cv2.circle(seedThresh, coord, 2, (0, 255, 0), cv2.FILLED, cv2.LINE_AA)

    
    return seedThresh




def sobel(frame):
    frame = cv2.GaussianBlur(frame,(5,5),0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    ksize = 3    
    grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=ksize, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=ksize, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    grad = cv2.addWeighted(abs_grad_y, 0.5, abs_grad_x, 0.5, 0)
    
    return grad


def executeVideo():
    fps = 150
    startTime = time.time()
    # for path in sorted(glob.glob('inputs/videos/*.mp4'), reverse=True):
    path = 'inputs/videos/' + sys.argv[1]
    print(path)
    cap = cv2.VideoCapture(path)
    ret, frame = cap.read()
    globalMean = np.array(cv2.mean(frame)[0:3]).astype(np.int)
    if sys.argv[1] == 'road.mp4':
        rollAvg = np.float32(cv2.resize(frame, (0, 0), None, 0.5, 0.5))
    else:
        rollAvg = np.float32(frame)

    while(cap.isOpened()):
        ret, frame = cap.read()
        
        while time.time() - startTime < 1/fps:
            True
        startTime = time.time()
        if not ret:
            break
        if sys.argv[1] == 'road.mp4':
            frame = cv2.resize(frame, (0, 0), None, 0.5, 0.5)
        cv2.imshow('frame', frame)
        out = sobel(frame)
        ret, out = cv2.threshold(out, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        kernel = (5,5)
        out = cv2.erode(out,kernel,iterations = 1)  
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
        out = getLanes(out)
        output = cv2.addWeighted(out, 0.5, frame, 0.5, 0)
        cv2.imshow('sobel', output)
        cv2.imshow('im', out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit(0)
    cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    executeVideo()

    