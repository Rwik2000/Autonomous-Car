import cv2
import numpy as np
import random
def floodfill(image):
    
    image=cv2.GaussianBlur(image,(7,7),0)
    height, width, _ = image.shape
    val=2
    loDiff=(val, val, val, val)
    upDiff=(val, val, val, val)

    seed_points=15
    w_offset=300
    for i in range(seed_points):
        seed=(int(width/2)-w_offset+int(w_offset*2/seed_points)*i, height - 100)
        # print(seed)
        cv2.floodFill(image, None, seedPoint=seed, newVal=(255, 255, 255), loDiff=loDiff, upDiff=upDiff)
        cv2.circle(image, seed, 2, (0, 255, 0), cv2.FILLED, cv2.LINE_AA)

    return image

video=cv2.VideoCapture('inputs/videos/athletics1.mp4')
# video=cv2.VideoCapture('./data/test photos and videos/road_traffic_2.mp4')

while(video.isOpened()):
    ret,image=video.read()
    if ret==True:
        try:            
            cv2.imshow('output',floodfill(image))
            # print('yo')
        except:
            cv2.imshow('output',image)
        if cv2.waitKey(1) == ord('q'):
            break
    else:
        break

video.release()
cv2.destroyAllWindows()
