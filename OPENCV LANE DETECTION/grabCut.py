## PV, DNE
import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob
import math
import time
import random
from scipy.spatial import distance


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



def floodFill(img):
    global seedPoints
    height, width, _ = img.shape
    orig = img.copy()
    # img, orig = floodfillCustomSeed(img, orig, ((int(width/2),  30)), color = (0,0,0), val =3)
    # img, orig = floodfillCustomSeed(img, orig, ((int(width/2) - random.randint(25,50), random.randint(25,50))),  color = (0,0,0), val =3)
    # img, orig = floodfillCustomSeed(img, orig, ((int(width/2) + random.randint(25,50), random.randint(25,50))),  color = (0,0,0), val =3)
    # cv2.imshow('img,', img)
    
    mask = np.zeros((height+2,width+2),np.uint8)
    
    for x in range(-100, 100, 30):
        img, orig, tempMask = floodfillCustomSeed(img, orig, ((int(width/2) - x, height - 75)))
        mask += tempMask
    mask[mask!=0] = 255
    tempmask = mask[0:height, 0:width]
    mean = cv2.mean(img, tempmask)[0:3]
    mean = (int(mean[0]), int(mean[1]), int(mean[2]))
    max_dist = 50

    
    colors = np.array([img[(int(width/2), height - 75)]])
    dist = distance.cdist(colors, img.reshape(-1, 3), 'euclidean')
    maska = np.any(dist <= max_dist, axis=0).reshape(img.shape[0], img.shape[1])
    img = np.repeat(maska[..., None], 3, axis=2) * img
    cv2.imshow('imga', img)
    # image_masked = img[np.where(tempmask)]
    # print(mean,np.mean(image_masked), np.var(image_masked[0]))
    # print(mean)
    if len(seedPoints) != 0:
        for x in seedPoints:
            pixel = np.array(img[seedPoints[0][::-1]])
            dist = ((pixel[0] - mean[0])**2 + (pixel[1] - mean[1])**2 + (pixel[2] - mean[2])**2)**(0.5)
            if dist > 50:
                break
                       
            
            # print(img.shape)
            
            img, orig, tempMask = floodfillCustomSeed(img, orig, x)
            mask += tempMask


        startx = seedPoints[0][0]
        endx = int(width/2)
        starty = seedPoints[0][1]
        endy =  height - 75
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
            dist = ((pixel[0] - mean[0])**2 + (pixel[1] - mean[1])**2 + (pixel[2] - mean[2])**2)**(0.5)
            if dist > 50:
                continue
            img, orig, tempMask = floodfillCustomSeed(img, orig, coord)
            mask += tempMask

    
    # img, orig = floodfillCustomSeed(img, orig, ((int(width/2) - random.randint(25,50), height - random.randint(25,50))))
    # img, orig = floodfillCustomSeed(img, orig, ((int(width/2) + random.randint(25,50), height - random.randint(25,50))))

    # img, orig = floodfillCustomSeed(img, orig, ((int(width/2) - random.randint(25,50), height - random.randint(25,50))))
    # img, orig = floodfillCustomSeed(img, orig, ((int(width/2) + random.randint(25,50), height - random.randint(25,50))))


    return img, orig, mask

def closest_node(node, nodes):
    closest_index = distance.cdist([node], nodes).argmin()
    return nodes[closest_index]


def grabCut(path = None, video = False, img = None, prevImg = None):
    global seedPoints
    start = time.time()
    h, w, _ = img.shape
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
    
    # img = kmeans_color_quantization(img, clusters=3)
    # cv2.imshow('asdf', kmeans)
    # img = kmeans_color_quantization(img, clusters = 5)
    # img = cv2.blur(img, (3,3))
    cv2.imshow('kmea', img)
    img, orig , mask= floodFill(img)

    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale
    
    # blur = cv2.blur(gray, (3, 3)) # blur the image
    # # cv2.imshow('flood', blur)
    # ret, thresh = cv2.threshold(blur, 250, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5,5),np.uint8)
    thresh = cv2.dilate(mask,kernel,iterations = 10)  
    thresh = cv2.erode(thresh,kernel,iterations = 10)  
    
    thresh = cv2.GaussianBlur(thresh,(21,21),0)
    ret, thresh = cv2.threshold(thresh, 100, 255, cv2.THRESH_BINARY)
    cv2.imshow('thresh', thresh)
    
    
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
    if not video:
        init = orig


    if init is None:
        drawing = init
    else:
        drawing = cv2.addWeighted(threshThreeChannel, 1, init, 1, 0)  


        


    # print(time.time() - start)
    return stackAndShow(orig, drawing, 'window', wait = not video)





def executeVideo():
    fps = 30
    startTime = time.time()
    for path in sorted(glob.glob('inputs/videos/*.mp4'), reverse=True):
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        rollAvg = np.float32(cv2.resize(frame, (0, 0), None, 0.5, 0.5))
        # rollAvg = np.float32(frame)


        while(cap.isOpened()):
            ret, frame = cap.read()
            while time.time() - startTime < 1/fps:
                True
            startTime = time.time()
            if not ret:
                break
            frame = cv2.resize(frame, (0, 0), None, 0.5, 0.5)
            cv2.accumulateWeighted(frame,rollAvg,0.1)
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

    