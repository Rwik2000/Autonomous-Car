import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob

import time
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

def stackAndShow(a, b, name,  wait = False):
    horiz = np.vstack((a, b))
    cv2.imshow(name, horiz)
    if wait:
        return cv2.waitKey(0)



def floodfillCustomSeed(img, orig, seed, color = (255,255,255), val = 1):
    loDiff=(val, val, val, val)
    upDiff=(val, val, val, val)
    
    cv2.floodFill(img, None, seedPoint=seed, newVal=color, loDiff=loDiff, upDiff=upDiff)
    cv2.circle(orig, seed, 2, (0, 255, 0), cv2.FILLED, cv2.LINE_AA)
    return img, orig

def floodFill(img):
    height, width, _ = img.shape
    orig = img.copy()
    # img, orig = floodfillCustomSeed(img, orig, ((int(width/2),  30)), color = (0,0,0), val =3)
    # img, orig = floodfillCustomSeed(img, orig, ((int(width/2) - random.randint(25,50), random.randint(25,50))),  color = (0,0,0), val =3)
    # img, orig = floodfillCustomSeed(img, orig, ((int(width/2) + random.randint(25,50), random.randint(25,50))),  color = (0,0,0), val =3)
    # cv2.imshow('img,', img)

    for x in range(-100, 100, 30):
        img, orig = floodfillCustomSeed(img, orig, ((int(width/2) - x, height - 30)))
    # img, orig = floodfillCustomSeed(img, orig, ((int(width/2) - random.randint(25,50), height - random.randint(25,50))))
    # img, orig = floodfillCustomSeed(img, orig, ((int(width/2) + random.randint(25,50), height - random.randint(25,50))))

    # img, orig = floodfillCustomSeed(img, orig, ((int(width/2) - random.randint(25,50), height - random.randint(25,50))))
    # img, orig = floodfillCustomSeed(img, orig, ((int(width/2) + random.randint(25,50), height - random.randint(25,50))))


    return img, orig


def grabCut(path = None, video = False, img = None, prevImg = None):
    
    start = time.time()
    if not video:
        img = cv2.imread(path)
        img = cv2.resize(img, (0, 0), None, 2, 2)
        orig = img.copy()
    else:
        # prevImg = cv2.resize(img, (0, 0), None, 0.5, 0.5)
        # img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
        # img = cv2.addWeighted(prevImg, 0.5, img, 0.5, 0)
        #  img = cv2.subtract(img, prevImg)
        init = prevImg.copy()
    
    # kmeans = kmeans_color_quantization(img, clusters=8)
    # cv2.imshow('asdf', kmeans)
    img = cv2.GaussianBlur(img,(3,3),0)


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
    color_contours = (0, 255, 0) # green - color for contours
    color = (255, 0, 0) # blue - color for convex hull
    for i in range(len(contours)):
        # draw ith contour
        cv2.drawContours(drawing, contours, i, color_contours, 3, 8, hierarchy)
        # draw ith convex hull object
        cv2.drawContours(drawing, hull, i, color, 3, 8)
    
    threshThreeChannel = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

    if not video:
        init = orig

    drawing = cv2.addWeighted(init, 1, drawing, 1, 0)
    drawing = cv2.addWeighted(threshThreeChannel, 0.5, drawing, 1, 0)


    print(time.time() - start)
    return stackAndShow(orig, drawing, 'window', wait = not video)

def executeVideo():
    fps = 15
    startTime = time.time()
    for path in sorted(glob.glob('inputs/videos/*'), reverse=True):
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        rollAvg = np.float32(cv2.resize(frame, (0, 0), None, 0.5, 0.5))
        while(cap.isOpened()):
            ret, frame = cap.read()
            while time.time() - startTime < 1/fps:
                True
            startTime = time.time()
            if not ret:
                break
            frame = cv2.resize(frame, (0, 0), None, 0.5, 0.5)
            cv2.accumulateWeighted(frame,rollAvg,0.08)
            result = cv2.convertScaleAbs(rollAvg)
            grabCut(img = result,  video = True, prevImg = frame)


           
            cv2.imshow('avg1',result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit(0)
        cv2.waitKey(0)
        cap.release()
        cv2.destroyAllWindows()

def executeImage():
    for path in sorted(glob.glob('inputs/images/*'), reverse=True):
        
        returnval = grabCut(path = path)
        
        if returnval == ord('q'):
            break

if __name__ == "__main__":
    executeVideo()
    # executeImage()

    