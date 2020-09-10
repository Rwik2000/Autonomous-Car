
# Python porgram to demonstrate 
# meanshift  
  
  
import numpy as np 
import cv2 
import sys
import time
  
# read video 
path = 'inputs/videos/' + sys.argv[1]
cap = cv2.VideoCapture(path) 
   
# retrieve the very first  
# frame from the video 
_, frame = cap.read() 
# frame = cv2.resize(frame, (0,0), None, 0.5, 0.5)
   
# set the region for the 
# tracking window p, q, r, s 
# put values according to yourself 
p, q, r, s = 150, 150, 460, 100
track_window = (r, p, s, q) 
   
      
# create the region of interest 
r_o_i = frame[p:p + q, r:r + s] 
  
# converting BGR to HSV format 
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
   
# apply mask on the HSV frame 
mask = cv2.inRange(hsv,  
                   np.array((0., 61., 33.)), 
                   np.array((180., 255., 255.))) 
  
# get histogram for hsv channel 
roi = cv2.calcHist([hsv], [0], mask,  
                   [180], [0, 180]) 
  
# normalize the retrieved values 
cv2.normalize(roi, roi, 0, 255,  
              cv2.NORM_MINMAX) 
   
# termination criteria, either 15  
# iteration or by at least 2 pt 
termination = (cv2.TERM_CRITERIA_EPS |  
               cv2.TERM_CRITERIA_COUNT 
               , 15, 2 ) 
fps = 30
while(True): 
    startTime = time.time()
    _, frame = cap.read() 
    # frame = cv2.resize(frame, (0,0), None, 0.5, 0.5)
   
    # frame = cv2.resize(frame,  
    #                    (1280, 720),  
    #                    fx = 0,  
    #                    fy = 0,  
    #                    interpolation = cv2.INTER_CUBIC) 
   
    # convert BGR to HSV format 
    hsv = cv2.cvtColor(frame,  
                       cv2.COLOR_BGR2HSV) 
      
    bp = cv2.calcBackProject([hsv],  
                             [0],  
                             roi,  
                             [0, 180],  
                             1) 
   
    # applying meanshift to get the new region 
    _, track_window = cv2.meanShift(bp,  
                                    track_window,  
                                    termination) 
   
    # Draw track window on the frame 
    x, y, w, h = track_window 
    vid = cv2.rectangle(frame, (x, y),  
                        (x + w, y + h),  
                        255, 2) 
      
    # show results 
    cv2.imshow('tracker', vid) 
    while time.time() - startTime < 1/fps:
        True
    k = cv2.waitKey(1) & 0xff
    if k == ord('q'): 
        break
          
# release cap object 
cap.release() 
  
# destroy all opened windows 
cv2.destroyAllWindows() 