import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob
import random
def kmeans_color_quantization(image, clusters=8, rounds=1):
    h, w = image.shape[:2]
    samples = np.zeros([h*w,3], dtype=np.float32)
    count = 0

    for x in range(h):
        for y in range(w):
            samples[count] = image[x][y]
            count += 1

    compactness, labels, centers = cv2.kmeans(samples,
            clusters, 
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001), 
            rounds, 
            cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    return res.reshape((image.shape))

def stackAndShow(a, b):
    horiz = np.vstack((a, b))
    cv2.imshow('image', horiz)
    return cv2.waitKey(0)

def floodFill(img):
    height, width, _ = img.shape
    orig = img.copy()
    seed = (int(width/2), height - 10)
    val  = 2
    loDiff=(val, val, val, val)
    color = (255,255,255)
    upDiff=(val, val, val, val)


    cv2.floodFill(img, None, seedPoint=seed, newVal=color, loDiff=loDiff, upDiff=upDiff)
    cv2.circle(orig, seed, 2, (0, 255, 0), cv2.FILLED, cv2.LINE_AA)


    seed = (int(width/2) - random.randint(0,50), height - random.randint(0,50))
    cv2.floodFill(img, None, seedPoint=seed, newVal=color, loDiff=loDiff, upDiff=upDiff)
    cv2.circle(orig, seed, 2, (0, 255, 0), cv2.FILLED, cv2.LINE_AA)

    seed = (int(width/2) + random.randint(0,50), height - random.randint(0,50))
    
    cv2.floodFill(img, None, seedPoint=seed, newVal=color, loDiff=loDiff, upDiff=upDiff)
    cv2.circle(orig, seed, 2, (255, 255, 0), cv2.FILLED, cv2.LINE_AA)
    return img, orig


def grabCut(path):
    start = time.time()
    img = cv2.imread(path)
    img = cv2.resize(img, (0, 0), None, 2, 2)
    orig = img.copy()
    # # img = kmeans_color_quantization(img, clusters=3)
    # cv2.imshow('asdf', img)
    
    height, width, _ = orig.shape
    kernel = np.ones((5,5),np.float32)/25
    img = cv2.filter2D(img,-1,kernel)

    # cv2.rectangle(orig,(0,int(height/2)),(width,int(height/2)), (255,0,0), 2)


    

    # bgdModel = np.zeros((1,65),np.float64)
    # fgdModel = np.zeros((1,65),np.float64)

    # # rect = (0,int(height/2),width,int(height/2))
    # rect = (0,0,width-1,height-1)
    # cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

    # mask2 = np.where((mask==2)|(mask==0),0,255).astype('uint8')
    # print(len(mask2[mask2 == 1]), len(mask2[0])*len(mask2), mask2.shape)


    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,3,1)
    # thresh = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)


    img, orig = floodFill(img)

    # cv2.imshow('flood', img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale
    
    blur = cv2.blur(gray, (3, 3)) # blur the image
    # cv2.imshow('flood', blur)
    ret, thresh = cv2.threshold(blur, 250, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5,5),np.uint8)
    thresh = cv2.dilate(thresh,kernel,iterations = 5)  
    thresh = cv2.erode(thresh,kernel,iterations = 5)  
    


    cv2.imshow('thresh', thresh)
    
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # create hull array for convex hull points
    hull = []

    # calculate points for each contour
    for i in range(len(contours)):
        # creating convex hull object for each contour
        hull.append(cv2.convexHull(contours[i], False))

    drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)

    for i in range(len(contours)):
        color_contours = (0, 255, 0) # green - color for contours
        color = (255, 0, 0) # blue - color for convex hull
        # draw ith contour
        cv2.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
        # draw ith convex hull object
        cv2.drawContours(drawing, hull, i, color, 1, 8)
    print(time.time() - start)
    return stackAndShow(orig, drawing)

import time
if __name__ == "__main__":
    for path in sorted(glob.glob('inputs/*'), reverse=True):
        
        returnval = grabCut(path)
        
        if returnval == ord('q'):
            break