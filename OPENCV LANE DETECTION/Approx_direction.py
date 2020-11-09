import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skimage import filters
def white_mask(image):
    hsv = cv2.cvtColor('image', cv2.COLOR_BGR2HSV)
    lower_green = np.array([0,0,150])
    upper_green = np.array([180,25,255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    res = cv2.bitwise_and(image,image, mask= mask)
    return res

def left_lane_kernel(image):
    kernel=[[0,-1,1],[0,-1,1],[0,-1,1]]
    kernel=np.array(kernel)
    kernel=kernel/2
    kernel=np.array(kernel)
    # print(kernel)
    img = cv2.filter2D(image, 0, kernel)
    return img

def right_lane_kernel(image):
    kernel=[[1,-1,0],[1,-2,0],[1,-1,0]]
    kernel=np.array(kernel)
    kernel=kernel/2
    kernel=np.array(kernel)
    img = cv2.filter2D(image, 0, kernel)
    return img
def resize(image):
    # print(image.shape)
    if(len(image.shape)==3):
        height,width,_=image.shape
    else:
        height,width=image.shape
    image=cv2.resize(image,(int(width/2),int(height/2)))
    return image
def predict(coordinates,mask):   

    
    xdata = coordinates[0][:,1]
    ydata = coordinates[0][:,0]


    z = np.polyfit(xdata, ydata, 5)
    f = np.poly1d(z)
    # print(min(coordinates[0][:,1]), max(coordinates[0][:,1]))
    t = np.arange(min(coordinates[0][:,1]), max(coordinates[0][:,1]), 1)
    # print(f(t))
    plt.plot(t, f(t))
    

def lane_separation(mask):
    right_lane=right_lane_kernel(mask)
    retval,right_lane = cv2.threshold(right_lane, 100, 255, cv2.THRESH_BINARY) 

    left_lane=left_lane_kernel(mask)
    retval,left_lane = cv2.threshold(left_lane, 100, 255, cv2.THRESH_BINARY) 
    indices = np.where(right_lane==255)
    coordinates_right = np.dstack((indices[0],indices[1]))
    
    # print(len(coordinates[0]))
    indices = np.where(left_lane==255)
    coordinates_left = np.dstack((indices[0],indices[1]))
    return left_lane,right_lane,coordinates_left,coordinates_right

def predict_ext_right(coordinates):
    bottom_point=np.zeros(2)
    top_point=np.zeros(2)
    top=min(coordinates[:,1])
    bot=max(coordinates[:,1])
    # print()
    for i in range(len(coordinates)):
        if coordinates[i][1]==top:
            if top_point[1]==0:
                top_point=coordinates[i]
            else:
                if coordinates[i][0]>top_point[0]:
                    top_point=coordinates[i]
        
        if coordinates[i][1]==bot:
            if top_point[1]==0:
                bottom_point=coordinates[i]
            else:
                if coordinates[i][0]>bottom_point[0]:
                    bottom_point=coordinates[i]
    
    return bottom_point,top_point

def predict_ext_left(coordinates):
    bottom_point=np.zeros(2)
    top_point=np.zeros(2)
    x=min(coordinates[:,1])
    y=max(coordinates[:,1])
    # print()
    for i in range(len(coordinates)):
        if coordinates[i][1]==x:
            if top_point[1]==0:
                top_point=coordinates[i]
            else:
                if coordinates[i][0]<top_point[0]:
                    top_point=coordinates[i]
        
        if coordinates[i][1]==y:
            if top_point[1]==0:
                bottom_point=coordinates[i]
            else:
                if coordinates[i][0]<bottom_point[0]:
                    bottom_point=coordinates[i]
    
    return bottom_point,top_point

def floodfill(image):
    input_img=image.copy()
    # input_img=cv2.cvtColor(input_img,cv2.COLOR_RGB2GRAY)
    input_img=cv2.GaussianBlur(input_img,(11,7),0)
    og_img=input_img.copy()

    height, width,_ = og_img.shape
    val=1
    loDiff=(val, val, val, val)
    upDiff=(val, val, val, val)

    seed_points=4
    w_offset=30
    floodflags = 4
    floodflags |= cv2.FLOODFILL_MASK_ONLY
    floodflags |= (255 << 8)
    mask = np.zeros((height+2,width+2),np.uint8)    
    for i in range(seed_points):
        for j in range(2):
            seed=(int(width/2)+(-1)**(j)*int(w_offset)*i, height - 100)
            num,image,mask,rect = cv2.floodFill(og_img, mask, seed, (0,0,0), loDiff=loDiff, upDiff=upDiff, flags=floodflags)
            cv2.circle(image, seed, 2, (0, 255, 0), cv2.FILLED, cv2.LINE_AA) 
    
    thickness = -1
    mask = cv2.rectangle(mask, (0,height-100), (width+2,height+2), (0,0,0), thickness) 
    
    kernel = np.ones((37,37),np.uint8)
    mask = cv2.dilate(mask,kernel,0)

    try:
        left_lane,right_lane,coordinates_left,coordinates_right=lane_separation(mask)

        bot_right,top_right=predict_ext_right(coordinates_right[0])
        top_left,bot_left=predict_ext_left(coordinates_left[0])

        bot_middle=(max(bot_left[0],bot_right[0]),int((bot_left[1]+bot_right[1])/2))
        top_middle=(min(top_left[0],top_right[0]),int((top_left[1]+top_right[1])/2))

        mask=cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
        cv2.circle(mask, (bot_right[1],bot_right[0]), 5, (255, 255, 0), cv2.FILLED, cv2.LINE_AA) 
        cv2.circle(mask, (top_right[1],top_right[0]), 5, (0, 255, 255), cv2.FILLED, cv2.LINE_AA)
        cv2.circle(mask, (top_left[1],top_left[0]), 5, (0, 255, 255), cv2.FILLED, cv2.LINE_AA)
        cv2.circle(mask, (bot_left[1],bot_left[0]), 5, (255, 255, 0), cv2.FILLED, cv2.LINE_AA)
        cv2.circle(mask, (bot_middle[1],bot_middle[0]), 5, (255, 0, 0), cv2.FILLED, cv2.LINE_AA)
        cv2.circle(mask, (top_middle[1],top_middle[0]), 5, (255, 0, 0), cv2.FILLED, cv2.LINE_AA) 
        
        # print(len(coordinates[0]))

        # cv2.imshow("canny",resize(canny))
        cv2.imshow("mask",resize(mask))
        cv2.imshow("left_lane",resize(left_lane))
        cv2.imshow("right_lane",resize(right_lane))
        cv2.imshow("normal",resize(image))
    except:
        cv2.imshow("mask",resize(mask))
        cv2.imshow("normal",resize(image))
    # predict(coordinates_left,left_lane)
    # predict(coordinates_right,right_lane)
    # plt.show()
    return image   


video=cv2.VideoCapture('inputs/videos/athletics1.mp4')
ret, frame = video.read()
prev = frame.copy()
while(video.isOpened()):
    ret,image=video.read()

    
    if ret==True:
        runAvg=image.copy()
        prev=image.copy()
        floodfill(runAvg)
        key = cv2.waitKey(42)
        if key == ord('q'):
            break
        if key == ord('p'):
            cv2.waitKey(-1) 
    else:
        break
    

video.release()
cv2.destroyAllWindows()
