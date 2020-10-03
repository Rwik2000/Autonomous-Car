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
from cleanSobel import LaneDetection


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

def stackAndShow(a, b, name):
    horiz = np.vstack((a, b))
    cv2.imshow(name, horiz)




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


# globalMean =  np.array([0,0,0])
nFrames = 1
def floodFill(rollingImg, currentFrame):
    global seedPoints, globalMean, nFrames
    height, width, _ = rollingImg.shape
    orig = rollingImg.copy()
    mask = np.zeros((height+2,width+2),np.uint8)
    variationMax = 50
    for x in range(-100, 100, 30):
        rollingImg, orig, tempMask = floodfillCustomSeed(rollingImg, orig, ((int(width/2) - x, height - heightStart - int(abs(x)/10))))
        mask += tempMask
    kernel = np.ones((5,5),np.uint8)
    mask[mask!=0] = 255
    if len(seedPoints) != 0:
        for x in seedPoints:
            pixel = np.array(rollingImg[seedPoints[0][::-1]])
            dist = ((pixel[0] - globalMean[0])**2 + (pixel[1] - globalMean[1])**2 + (pixel[2] - globalMean[2])**2)**(0.5)
            if dist > variationMax:
                break
            rollingImg, orig, tempMask = floodfillCustomSeed(rollingImg, orig, x)
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
            pixel = rollingImg[coord[::-1]]
            dist = ((pixel[0] - globalMean[0])**2 + (pixel[1] - globalMean[1])**2 + (pixel[2] - globalMean[2])**2)**(0.5)
            if dist > variationMax:
                continue
            rollingImg, orig, tempMask = floodfillCustomSeed(rollingImg, orig, coord)
            mask += tempMask

    
    finalMask = mask.copy()
    finalMask = cv2.dilate(finalMask,kernel,iterations = 1) 
    finalMask = cv2.erode(finalMask,kernel,iterations = 1)  
    
    # cv2.imshow('seededmask', t)    


    mask[mask!=0] = 255
    tempmask = mask[0:height, 0:width]
    mean = cv2.mean(rollingImg, tempmask)[0:3]
    mean = np.array([int(mean[0]), int(mean[1]), int(mean[2])])
    globalMean *= nFrames
    globalMean += mean
    nFrames += 1
    globalMean = np.array([int(globalMean[0]/nFrames), int(globalMean[1]/nFrames), int(globalMean[2]/nFrames) ])
    if nFrames >= 200:
        nFrames = 50
    max_dist = 50
    colors = np.array([globalMean])
    tempImg = currentFrame.copy()
    dist = distance.cdist(colors, tempImg.reshape(-1, 3), 'euclidean')
    maska = np.any(dist <= max_dist, axis=0).reshape(tempImg.shape[0], tempImg.shape[1])
    tempImg = np.repeat(maska[..., None], 3, axis=2) * tempImg
    gray = cv2.cvtColor(tempImg, cv2.COLOR_BGR2GRAY)
    
    finalMask = cv2.erode(gray,kernel,iterations = 1)  
    finalMask = cv2.dilate(finalMask,kernel,iterations = 1) 


    ret, t1 = cv2.threshold(finalMask, 1, 255, cv2.THRESH_BINARY)
    finalMask = cv2.cvtColor(t1, cv2.COLOR_GRAY2RGB)
    return rollingImg, orig, mask, finalMask

def closest_node(node, nodes):
    closest_index = distance.cdist([node], nodes).argmin()
    return nodes[closest_index]


def floodFillMethod(video = False, rollingImage = None, currentFrame = None):
    global seedPoints
    start = time.time()


    h, w, _ = rollingImage.shape

    # prevImg = cv2.resize(img, (0, 0), None, 0.5, 0.5)
    # img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
    # img = cv2.addWeighted(prevImg, 0.5, img, 0.5, 0)
    #  img = cv2.subtract(img, prevImg)
    init = currentFrame.copy()

    # img = kmeans_color_quantization(img, clusters=3)
    # cv2.imshow('asdf', kmeans)
    # img = kmeans_color_quantization(img, clusters = 5)
    # img = cv2.blur(img, (3,3))

    
    rollingImage, orig , mask, finalMask= floodFill(rollingImage, currentFrame)
    # cv2.imshow('init', img)

    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale
    
    # blur = cv2.blur(gray, (3, 3)) # blur the image
    # # cv2.imshow('flood', blur)
    # ret, thresh = cv2.threshold(blur, 250, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5,5),np.uint8)
    thresh = cv2.dilate(mask,kernel,iterations = 5)  
    thresh = cv2.erode(thresh,kernel,iterations = 5)  
    
    thresh = cv2.GaussianBlur(thresh,(21,21),0)
    ret, thresh = cv2.threshold(thresh, 100, 255, cv2.THRESH_BINARY)
    # cv2.imshow('thresh', thresh)
    
    
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:

        # perimeter = cv2.arcLength(cnt,True)
        # epsilon = 0.005*cv2.arcLength(cnt,True)
        # approx = cv2.approxPolyDP(cnt,epsilon,True)
        # cv2.drawContours(init, [approx], -1, (255,192,203), cv2.FILLED, 8, )
        # M = cv2.moments(cnt)
        # cX = int(M["m10"] / M["m00"])
        # cY = int(M["m01"] / M["m00"])
        center = (int(w/2), int(h/2))
        # print(cnt.shape)
        # extTop = closest_node(center, cnt)
        # print(cv2.contourArea(cnt))
        extTop = tuple(cnt[cnt[:, :, 1].argmin()][0])
        extTop = (min(extTop[0], w - 1), min(extTop[1] -1, h-1))
        if extTop[0] < 0 or extTop[1] < 0:
            # print(extTop)
            extTop = (0,0)
        # print(extTop)
        seedPoints = [ extTop]
        # print(seedPoints)
        # cv2.circle(thresh, extTop, 7, (120, 0, 255), -1)
        # rows,cols = img.shape[:2]
        # [vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
        # lefty = int((-x*vy/vx) + y)
        # righty = int(((cols-x)*vy/vx)+y)
        # cv2.line(init,(cols-1,righty),(0,lefty),(0,255,0),2)
    
    threshThreeChannel = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)[0:h, 0:w]
    # print(drawing.shape)
    # drawing = drawing[0:h, 0:w]
    # drawing = cv2.cvtColor(drawing, cv2.COLOR_GRAY2RGB)




        
    

    # print(time.time() - start)

    return finalMask


def sobel(frame):

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
    grad = cv2.cvtColor(grad, cv2.COLOR_GRAY2RGB)
    return grad

heightStart = 120
def executeVideo():
    global globalMean
    fps = 30
    startTime = time.time()
    # for path in sorted(glob.glob('inputs/videos/*.mp4'), reverse=True):
    path = 'inputs/videos/' + sys.argv[1]
    print(path)
    cap = cv2.VideoCapture(path)
    ret, frame = cap.read()
    globalMean = np.array(cv2.mean(frame)[0:3]).astype(np.int)
    if sys.argv[1] in ['road.mp4', 'cityroads.mp4']:
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
        if sys.argv[1] in ['road.mp4', 'cityroads.mp4']:
            frame = cv2.resize(frame, (0, 0), None, 0.5, 0.5)
        # frame = cv2.blur(frame, (3,3))
        # frame = cv2.resize(frame, (0,0), None, 0.1, 0.1)
        # frame = kmeans_color_quantization(frame, 8)
        # frame = cv2.pyrMeanShiftFiltering(frame, 2, 5)
        # frame = cv2.resize(frame, (0,0), None,10, 10)
        
        
        # result = cv2.convertScaleAbs(rollAvg)
        # kernel = np.ones((7,7),np.float32)/25
        # dst = cv2.filter2D(frame,-1,kernel)
        # out = sobel(dst)
        


        
        # cv2.imshow('sobel', out)
        cv2.accumulateWeighted(frame,rollAvg,0.2)
        result = cv2.convertScaleAbs(rollAvg)
        floodedMask = floodFillMethod(rollingImage = result, currentFrame = frame)
        
        sobelMask, _ = sobelFilt.detect(frame)
        
        netMask = floodedMask.copy()
        netMask[sobelMask != 0] = 0



        cv2.imshow('flood', floodedMask)
        cv2.imshow('sobel', sobelMask)


        stackAndShow(frame, netMask, 'window')
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit(0)
    cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()

def executeImage():
    for path in sorted(glob.glob('inputs/images/*'), reverse=True):
        
        returnval = floodFillMethod(path = path)
        
        if returnval == ord('q'):
            break
sobelFilt = LaneDetection(heightStart, 14, 20)
if __name__ == "__main__":

    executeVideo()
    # executeImage()

    