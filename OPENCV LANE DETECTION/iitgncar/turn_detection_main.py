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
from .turn_detection import lane_separation

seedPoints = []
def kmeans_color_quantization(image, clusters=8, rounds=1):
    h, w = image.shape[:2]
    samples = np.zeros([h*w,3], dtype=np.float32)
    count = 0

    for x in range(h):
        for y in range(w):
            samples[count] = image[x][y]
            count += 1

    _, labels, centers = cv2.kmeans(samples,
            clusters, 
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001), 
            rounds, 
            cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    return res.reshape((image.shape))

def stackAndShow(a, b, name,  wait = False):
    horiz = np.vstack((a, b))
    cv2.imshow(name, horiz)
    if wait:
        return cv2.waitKey(0)



def floodfillCustomSeed(img, orig, seed, color = (255,255,255), val = 1):
    loDiff=(val, val, val, val)
    upDiff=(val, val, val, val)
    floodflags = 4
    floodflags |= cv2.FLOODFILL_MASK_ONLY
    floodflags |= (255 << 8)
    h, w, _ = img.shape
    mask = np.zeros((h+2,w+2),np.uint8)
    cv2.floodFill(img, mask, seedPoint=seed, newVal=color, loDiff=loDiff, upDiff=upDiff, flags = floodflags)
    cv2.circle(orig, seed, 2, (0, 255, 0), cv2.FILLED, cv2.LINE_AA)
    return img, orig, mask

heightStart = 50
# globalMean =  np.array([0,0,0])
nFrames = 1
def floodFill(img, prevframe):
    global seedPoints, globalMean, nFrames
    height, width, _ = img.shape
    orig = img.copy()
    mask = np.zeros((height+2,width+2),np.uint8)
    variationMax = 50
    for x in range(-100, 100, 30):
        img, orig, tempMask = floodfillCustomSeed(img, orig, ((int(width/2) - x, height - heightStart - int(abs(x)/10))))
        mask += tempMask
    kernel = np.ones((5,5),np.uint8)
    mask[mask!=0] = 255

    t = cv2.erode(mask,kernel,iterations = 1)  
    t = cv2.dilate(t,kernel,iterations = 1)    
    
    # print(mean,np.mean(image_masked), np.var(image_masked[0]))
    # print(mean)
    if len(seedPoints) != 0:
        for x in seedPoints:
            pixel = np.array(img[seedPoints[0][::-1]])
            dist = ((pixel[0] - globalMean[0])**2 + (pixel[1] - globalMean[1])**2 + (pixel[2] - globalMean[2])**2)**(0.5)
            if dist > variationMax:
                break
            
            # print(img.shape)
            
            img, orig, tempMask = floodfillCustomSeed(img, orig, x)
            mask += tempMask


        startx = seedPoints[0][0]
        endx = int(width/2)
        starty = seedPoints[0][1]
        endy =  height - heightStart
        dist = 10


        theta = math.atan2((endy - starty),(endx - startx))
        delx = dist*math.cos(theta)
        dely = dist*math.sin(theta)


        totdist = math.sqrt((endy - starty)**2 + (endx - startx)**2)
        for i in range(0, int(totdist/dist)):

            coorda = int(min(startx + delx*i, width- 2))
            coordb =int( min(starty + dely*i, height - 2))
            
            coord = (coorda, coordb)
            # print(coord)
            # print(coord, slope, x, img.shape)
            pixel = img[coord[::-1]]
            dist = ((pixel[0] - globalMean[0])**2 + (pixel[1] - globalMean[1])**2 + (pixel[2] - globalMean[2])**2)**(0.5)
            if dist > variationMax:
                continue
            img, orig, tempMask = floodfillCustomSeed(img, orig, coord)
            mask += tempMask

    # cv2.imshow("hehe",img)
    mask[mask!=0] = 255
    tempmask = mask[0:height, 0:width]
    mean = cv2.mean(img, tempmask)[0:3]
    mean = np.array([int(mean[0]), int(mean[1]), int(mean[2])])
    globalMean *= nFrames
    globalMean += mean
    nFrames += 1
    globalMean = np.array([int(globalMean[0]/nFrames), int(globalMean[1]/nFrames), int(globalMean[2]/nFrames) ])
    if nFrames >= 200:
        nFrames = 10
    # print(globalMean)
    max_dist = 50
    
    colors = np.array([globalMean])
    dist = distance.cdist(colors, prevframe.reshape(-1, 3), 'euclidean')
    maska = np.any(dist <= max_dist, axis=0).reshape(prevframe.shape[0], prevframe.shape[1])
    prevframe = np.repeat(maska[..., None], 3, axis=2) * prevframe
    gray = cv2.cvtColor(prevframe, cv2.COLOR_BGR2GRAY)
    

    # t = cv2.blur(gray, (5,5)) 
    t = cv2.erode(gray,kernel,iterations = 1)  
    t = cv2.dilate(t,kernel,iterations = 1) 
    
    ret, t1 = cv2.threshold(t, 1, 255, cv2.THRESH_BINARY)
    t = cv2.cvtColor(t1, cv2.COLOR_GRAY2RGB)

    return img, orig, mask

def closest_node(node, nodes):
    closest_index = distance.cdist([node], nodes).argmin()
    return nodes[closest_index]

def grabCut(path = None, video = False, img = None, prevImg = None):
    if not video:
        global globalMean
        globalMean = np.array(cv2.mean(img)[0:3]).astype(np.int)
    global seedPoints
    start = time.time()
    h, w, _ = img.shape
    if not video:
        # img = cv2.imread(path)
        img = cv2.resize(img, (0, 0), None, 2, 2)
        orig = img.copy()
    else:
        init = prevImg.copy()
    
    img, orig , mask= floodFill(img, prevImg)

    kernel = np.ones((5,5),np.uint8)
    thresh = cv2.dilate(mask,kernel,iterations = 20)  
    thresh = cv2.erode(thresh,kernel,iterations = 20)  
    
    thresh = cv2.GaussianBlur(thresh,(21,21),0)
    ret, thresh = cv2.threshold(thresh, 100, 255, cv2.THRESH_BINARY)

    thresh1=cv2.resize(thresh,(int((w+2)/2),int((h+2)/2)))
    cv2.imshow('thresh', thresh1)
    
    
    # contours, hierarchy,_ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    m=cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours=m[1]
    # print(len(m))
    for cnt in contours:

        center = (int(w/2), int(h/2))
        extTop = tuple(cnt[cnt[:, :, 1].argmin()][0])
        extTop = (min(extTop[0], w - 1), min(extTop[1] -1, h-1))
        if extTop[0] < 0 or extTop[1] < 0:
            # print(extTop)
            extTop = (0,0)
        # print(extTop)
        seedPoints = [ extTop]
    values=lane_separation(thresh,img)
    return values[0]