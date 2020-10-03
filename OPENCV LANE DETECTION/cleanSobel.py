## PV, DNE
import numpy as np
import cv2


class LaneDetection:
    def __init__ (self, heightStart, heightSkipInterval, nSeeds):
        self.heightStart = heightStart
        self.heightSkipInterval = heightSkipInterval
        self.nSeeds = nSeeds
        self.rollAvgOutput = None
        self.rollAvg = None

    def getLanes(self, thresh):
        height = thresh.shape[0]
        width = thresh.shape[1]
        seedThresh = np.zeros_like(thresh)
        # seedThresh = thresh.copy()
        leftLaneIndex = None
        rightLaneIndex = None
        thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
        for i in range(0, self.nSeeds):
            coord = (int(width/2),height - self.heightSkipInterval*i -self.heightStart)
            row = thresh[coord[1]]
            isLane = (row != 0)
            
            startPos = 0
            if i != 0:
                startPos = int((leftLaneIndex[0] + rightLaneIndex[0])/2)
            leftLaneIndex = None
            rightLaneIndex = None
            for j in range(startPos, coord[0]):
                if leftLaneIndex != None and rightLaneIndex != None:
                    break
                if leftLaneIndex == None and isLane[coord[0] - j] == True:
                    leftLaneIndex = (int(width/2) - j,height - self.heightSkipInterval*i -self.heightStart)
                if rightLaneIndex == None and isLane[coord[0] + j] == True:
                    rightLaneIndex = (int(width/2) + j,height - self.heightSkipInterval*i -self.heightStart)
            if leftLaneIndex == None:
                leftLaneIndex = coord
            if rightLaneIndex == None:
                rightLaneIndex = coord
                
            cv2.circle(seedThresh, rightLaneIndex, 3, (0, 0, 255), cv2.FILLED, cv2.LINE_AA)
            cv2.circle(seedThresh, leftLaneIndex, 3, (255, 255, 0), cv2.FILLED, cv2.LINE_AA)
            # print(closest_node(coord, ))
            cv2.circle(seedThresh, coord, 2, (0, 255, 0), cv2.FILLED, cv2.LINE_AA)
        return seedThresh

    def sobel(self, frame):
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

    def postProcessFrame(self, image):
        ret, out = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        kernel = (5,5)
        out = cv2.dilate(out,kernel,iterations = 1)  
        out = cv2.erode(out,kernel,iterations = 1)  
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
        return out

    def detect(self, frame):
        

        sobelled = self.sobel(frame)
        if self.rollAvgOutput is None:
            self.rollAvgOutput = sobelled
        if self.rollAvg is None:
            self.rollAvg = sobelled.copy().astype(float)
        else:
            cv2.accumulateWeighted(sobelled, self.rollAvg, 0.2)
            self.rollAvgOutput = cv2.convertScaleAbs(self.rollAvg) 
            # cv2.imshow('accumulated', self.rollAvgOutput)
            # cv2.waitKey(1)

        out = self.postProcessFrame(sobelled)
        lanes = self.getLanes(out)
        return out, lanes

