import cv2
import numpy as np
import random
import time
import matplotlib.pyplot as plt
# from skimage.morphology import skeletonize
# from skimage import filters
# from numba import jit'
from bezier_mine import bezier_coordinates
import PIL
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
def polynom(points,mask):
    x_val=points[:,1]
    y_val=points[:,0]
    y_mid=int(min(y_val)+max(y_val)/2)
    x_curve=x_val[y_val<=y_mid]
    y_curve=y_val[y_val<=y_mid]
    # x_curve=x_val
    # y_curve=y_val
    x_st=x_val[y_val>=y_mid]
    y_st=y_val[y_val>=y_mid]

    #curve
    coeffs_curve=np.polyfit(x_curve, y_curve, 2, rcond=None, full=False, w=None, cov=False)
    x_curve=np.linspace(min(x_curve),max(x_curve),num=100)
    func_curve=np.poly1d(coeffs_curve)
    y_curve=func_curve(x_curve)
    plt.plot(x_curve, y_curve, label="deg=2")
    # plt.show()

    #straight
    coeffs_st=np.polyfit(x_st, y_st, 1, rcond=None, full=False, w=None, cov=False)
    x_st=np.linspace(min(x_st),max(x_st),num=100)
    func_st=np.poly1d(coeffs_st)
    y_st=func_st(x_st)
    plt.plot(x_st, y_st, label="deg=1")
    plt.show()
    return x_st,y_st,x_curve,y_curve

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
    bezier_coordinates(coordinates_left,coordinates_right,left_lane,10)

    # print(coordinates_left)
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
            seed=(int(width/2)+(-1)**(j)*int(w_offset)*i, height - 30)
            num,image,mask,rect = cv2.floodFill(og_img, mask, seed, (0,0,0), loDiff=loDiff, upDiff=upDiff, flags=floodflags)
            cv2.circle(image, seed, 2, (0, 255, 0), cv2.FILLED, cv2.LINE_AA) 
    
    # thickness = -1
    # mask = cv2.rectangle(mask, (0,height-40), (width+2,height+2), (0,0,0), thickness) 
    
    kernel = np.ones((37,37),np.uint8)
    mask = cv2.dilate(mask,kernel,0)


    # try:
    left_lane,right_lane,coordinates_left,coordinates_right=lane_separation(mask)
    
    # cv2.imshow("canny",resize(canny))
    cv2.imshow("mask",resize(mask))
    cv2.imshow("left_lane",left_lane)
    cv2.imshow("right_lane",right_lane)
    cv2.imshow("normal",resize(image))
    # except:
    #     cv2.imshow("mask",resize(mask))
    #     cv2.imshow("normal",resize(image))
    # predict(coordinates_left,left_lane)
    # predict(coordinates_right,right_lane)
    # plt.show()
    print("yo")
    return image   


# video=cv2.VideoCapture('challenge4.mp4')
# # video=cv2.VideoCapture('./data/test photos and videos/road_traffic_2.mp4')
# ret, frame = video.read()
# prev = frame.copy()
# while(video.isOpened()):
#     ret,image=video.read()

    
#     if ret==True:
#         runAvg=image.copy()
#         # runAvg=cv2.addWeighted(image,0.5,prev,0.5,0)
#         prev=image.copy()
#         floodfill(runAvg)
#         key = cv2.waitKey(42)
#         if key == ord('q'):
#             break
#         if key == ord('p'):
#             cv2.waitKey(-1) 
#     else:
#         break
    

# video.release()
# cv2.destroyAllWindows()
# start=time.time()
# input_img=cv2.imread("2.jpg")
# output_img=floodfill(input_img)
# end=time.time()
# print(end-start)
# cv2.waitKey(0)
