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
fps = 30
maxNum = 20
boundwidth = 10



def getLanes(thresh):
    height = thresh.shape[0]
    width = thresh.shape[1]
    # seedThresh = np.zeros_like(thresh)
    seedThresh = thresh.copy()
    thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    leftLanePoints = []
    rightLanePoints = []
    for i in range(0, maxNum):
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
            
            # leftLanePoints.append(True)
        else:
            leftLanePoints.append(leftLaneIndex)
            # cv2.circle(seedThresh, leftLaneIndex, 3, (255, 255, 0), cv2.FILLED, cv2.LINE_AA)

        if rightLaneIndex == None:
            rightLaneIndex = coord
            # rightLanePoints.append(True)
        else:
            rightLanePoints.append(rightLaneIndex)
            # cv2.circle(seedThresh, rightLaneIndex, 3, (0, 0, 255), cv2.FILLED, cv2.LINE_AA)



    variationMax = 40
    leftLanePointsTemp = []
    for i in range(len(leftLanePoints)):
        item = leftLanePoints[i]
        if distance.euclidean(item,leftLanePoints[i-1]) < variationMax : 
            leftLanePointsTemp.append(item)
            cv2.circle(seedThresh, item, 3, (255, 255, 0), cv2.FILLED, cv2.LINE_AA)
    leftLanePoints = leftLanePointsTemp
    leftedge = max(leftLanePoints[-1][0] - boundwidth, 0)
    rightedge = min(leftLanePoints[-1][0] + boundwidth, thresh.shape[1])
    topedge = max(leftLanePoints[-1][1] - boundwidth, 0)
    bottomedge = min(leftLanePoints[-1][1] + boundwidth, thresh.shape[0])
    temp = cv2.resize(thresh[topedge:bottomedge, leftedge:rightedge], (200, 200))
    cv2.imshow('window', temp)
    flingVelocity = np.array(leftLanePoints[-1]) - np.array(leftLanePoints[-2])
    maxxer = 30
    # print(flingVelocity, end=  ' ')
    if flingVelocity[0] < 0:
        flingVelocity[0] = max(flingVelocity[0], -maxxer)
    elif flingVelocity[0] > 0:
        flingVelocity[0] = min(flingVelocity[0], maxxer)
    if flingVelocity[1] < 0:
        flingVelocity[1] = max(flingVelocity[1], -maxxer)
    elif flingVelocity[1] > 0:
        flingVelocity[1] = min(flingVelocity[1], maxxer)
    # print(flingVelocity, end = ' ')

    endPoint = np.array(leftLanePoints[-1])
    currentPoint = endPoint
    for i in range(5):
        currentPoint[0] += flingVelocity[0]
        currentPoint[1] += flingVelocity[1]
        cv2.circle(seedThresh, tuple(currentPoint), 3, (255, 255, 0), cv2.FILLED, cv2.LINE_AA)



    variationMax = 40
    rightLanePointsTemp = []
    for i in range(len(rightLanePoints)):
        item = rightLanePoints[i]
        if distance.euclidean(item,rightLanePoints[i-1]) < variationMax : 
            rightLanePointsTemp.append(item)
            cv2.circle(seedThresh, item, 3, (0, 255, 255), cv2.FILLED, cv2.LINE_AA)
    rightLanePoints = rightLanePointsTemp
    rightedge = max(rightLanePoints[-1][0] - boundwidth, 0)
    rightedge = min(rightLanePoints[-1][0] + boundwidth, thresh.shape[1])
    topedge = max(rightLanePoints[-1][1] - boundwidth, 0)
    bottomedge = min(rightLanePoints[-1][1] + boundwidth, thresh.shape[0])
    # temp = cv2.resize(thresh[topedge:bottomedge, rightedge:rightedge], (200, 200))
    # cv2.imshow('windowright', temp)
    if len(rightLanePoints) < 2:
        return seedThresh


    flingVelocity = np.array(rightLanePoints[-1]) - np.array(rightLanePoints[-2])
    maxxer = 30
    print(flingVelocity, end=  ' ')
    if flingVelocity[0] < 0:
        flingVelocity[0] = max(flingVelocity[0], -maxxer)
    elif flingVelocity[0] > 0:
        flingVelocity[0] = min(flingVelocity[0], maxxer)
    if flingVelocity[1] < 0:
        flingVelocity[1] = max(flingVelocity[1], -maxxer)
    elif flingVelocity[1] > 0:
        flingVelocity[1] = min(flingVelocity[1], maxxer)
    print(flingVelocity)

    endPoint = np.array(rightLanePoints[-1])
    currentPoint = endPoint
    for i in range(5):
        currentPoint[0] += flingVelocity[0]
        currentPoint[1] += flingVelocity[1]
        cv2.circle(seedThresh, tuple(currentPoint), 3, (0, 255, 255), cv2.FILLED, cv2.LINE_AA)

    
    
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
    
    startTime = time.time()
    # for path in sorted(glob.glob('inputs/videos/*.mp4'), reverse=True):
    path = 'inputs/videos/' + sys.argv[1]
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

    