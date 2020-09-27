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


def executeVideo():
    path = 'inputs/videos/' + sys.argv[1]
    print(path)
    cap = cv2.VideoCapture(path)
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameSum = frame.astype(float)
    i = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # if i == 1000:
        #     frameSum = frame.astype(int)
        #     i = 1


        frameSum += frame.astype(int)*0.4
        
        frameMean = np.divide(frameSum, i)
        frameMean = frameMean.astype(np.uint8)
        # frameMean = cv2.cvtColor(frameMean, cv2.COLOR_BGR2GRAY)
        frameSub = frame.astype(np.uint8) - frameMean

        i = i + 1

        cv2.imshow('subd', frameSub)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit(0)
        
    cap.release()
    
    
            
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    executeVideo()
    # executeImage()

    