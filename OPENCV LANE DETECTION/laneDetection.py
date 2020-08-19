import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
from scipy import stats



def convHSV(img):
    return cv2.cvtColor(img,cv2.COLOR_RGB2HSV)

def convHSL(img):
    return cv2.cvtColor(img,cv2.COLOR_RGB2HLS)

def isolateYellowHSL(img):
    low_threshold=np.array([30,38,110],dtype=np.uint8)
    high_threshold=np.array([150,204,255],dtype=np.uint8)
    yellow=cv2.inRange(img,low_threshold,high_threshold)
    return yellow

def isolateWhiteHSL(img):
    low_threshold=np.array([0,195,0],dtype=np.uint8)
    high_threshold=np.array([180,255,255],dtype=np.uint8)
    white=cv2.inRange(img,low_threshold,high_threshold)
    return white

def MaskCombine(img,hsl_yellow,hsl_white):
    hsl_mask=cv2.bitwise_or(hsl_yellow,hsl_white)
    return cv2.bitwise_and(img,img,mask=hsl_mask)

def filter_hsl(img):
    hsl_img=convHSL(img)
    hsl_yellow=isolateYellowHSL(hsl_img)
    hsl_white=isolateWhiteHSL(hsl_img)
    return MaskCombine(img,hsl_yellow,hsl_white)


def grayscale(img):
    return cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)


def gaussianblur(grayscale_img):
    kernel=5
    return cv2.GaussianBlur(grayscale_img,(kernel,kernel),0)

def canny(img,low_threshold,high_threshold):
    return cv2.Canny(img,low_threshold,high_threshold)

def shape_of_img(img):
    imshape=img.shape
    height=imshape[0]
    width=imshape[1]
    vertices=None

#CHANGE: Set THESE PARAMETERS ACCORDING TO CAMERA ANGLE
    
    if(width,height)==(960,540):
        bottom_left=(130,img.shape[0]-1)
        top_left=(410,350)
        top_right=(650,350)
        bottom_right=(img.shape[1]-30,img.shape[0]-1)
        vertices=np.array([bottom_left,top_left,top_right,bottom_right])
        
    else:
        bottom_left=(200,680)
        top_left=(400,450)
        top_right=(750,450)
        bottom_right=(1100,650)
        vertices=np.array([bottom_left,top_left,top_right,bottom_right])
        
        
    return vertices

def ROI(img):
    mask=np.zeros_like(img)
    if len(img.shape)>2:
        channel_count=img.shape[2]
        ignore_mask_color=(255,) * channel_count
    else:
        ignore_mask_color=255

    vertices=shape_of_img(img)
    
    cv2.fillPoly(mask,np.int32([vertices]),ignore_mask_color)
    mask_img=cv2.bitwise_and(img,mask)
    return mask_img

def mask_bright(mask_img,img):
    return cv2.addWeighted(mask_img,1,img,1,0)

def hough_transform(img,rho,theta,threshold,min_line_len,max_line_gap):
    return cv2.HoughLinesP(img,rho,theta,threshold,np.array([]),min_line_len,max_line_gap)


def draw_lines(img,lines,color,thickness=10,make_copy=True):
    img_copy=np.copy(img) if make_copy else img

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img_copy,(x1,y1),(x2,y2),color,thickness)
    
    return img_copy

def separate_lines(lines,img):
    x_mid=img.shape[1]/2
    left_lanes=[]
    right_lanes=[]

    for line in lines:
        for x1,y1,x2,y2 in line:
            dx=x2-x1
            if dx==0:
                continue
            dy=y2-y1
            if dy==0:
                continue

            slope=dy/dx

            epsilon=0.01
            if abs(slope)<=epsilon:
                continue

            if slope<0 and x1<x_mid and x2<x_mid :
                left_lanes.append([[x1,y1,x2,y2]])
            elif x1>=x_mid and x2>=x_mid:
                right_lanes.append([[x1,y1,x2,y2]])
    return left_lanes,right_lanes

def color_lanes(img,left_lanes,right_lanes,left_color=[0,0,255],right_color=[255,0,0]):
    color_left_side=draw_lines(img,left_lanes,color=left_color,make_copy=True)
    color_right_side=draw_lines(img,right_lanes,color=right_color,make_copy=True)
    return cv2.addWeighted(color_left_side,0.5,color_right_side,0.7,0)

#LINES EXTRAPOLATION
def finding_lane_line(lines):
    xs=[]
    ys=[]

    for line in lines:
        for x1,y1,x2,y2 in line:
            xs.append(x1)
            xs.append(x2)
            ys.append(y1)
            ys.append(y2)
    slope,intercept,r_value,p_value,std_err=stats.linregress(xs,ys)
    return(slope,intercept)

def tracing_lane(img,lines,color,top_y,make_copy=True):
    A,b=finding_lane_line(lines)
    vertices=shape_of_img(img)
    bottom_y=img.shape[0]-1
    bottom_x=(bottom_y-b)/A
    top_y=top_y+300
    top_x=(top_y-b)/A

    new_lines=[[[int(bottom_x), int(bottom_y), int(top_x), int(top_y)]]]
    return draw_lines(img,new_lines,color,make_copy=make_copy)
def trace_both(img,left_lane,right_lane):
    vertices=shape_of_img(img)
    region_top_left=vertices[0][0]

    full_left=tracing_lane(img,left_lane,[0,0,255],region_top_left,make_copy=True)
    full_left_right=tracing_lane(full_left,right_lane,[255,0,0],region_top_left,make_copy=False)

    new_image=cv2.addWeighted(img,0.3,full_left_right,0.7,0)
    return new_image


def display_final_lane_lines(image):
    img=np.copy(image)
    img_new=np.copy(image)#this is used for hadling errors
    #so there are 2 copies of original images
    rho=1
    theta=(np.pi/180)*1
    threshold=15
    min_line_len=10
    max_line_gap=5

    try:
        img=gaussianblur(img)
        img=filter_hsl(img)
        img=grayscale(img)
        img=canny(img,50,150)
        img=ROI(img)
        
        hough_lines=hough_transform(img,rho,theta,threshold,min_line_len,max_line_gap)

        separated_lane_lines=separate_lines(hough_lines,image)
        left_lane_lines=separated_lane_lines[0]
        right_lane_lines=separated_lane_lines[1]
        diff_lane_color=trace_both(image,left_lane_lines,right_lane_lines)
        
        #return img
        return diff_lane_color
    except ValueError:
        
        img=img_new
        masked_img=ROI(img)
        img=mask_bright(masked_img,img)
        img=gaussianblur(img)
        img=filter_hsl(img)
        img=grayscale(img)
        img=canny(img,50,150)
        img=ROI(img)
        
        hough_lines=hough_transform(img,rho,theta,threshold,min_line_len,max_line_gap)
        separated_lane_lines=separate_lines(hough_lines,image)
        left_lane_lines=separated_lane_lines[0]
        right_lane_lines=separated_lane_lines[1]
        diff_lane_color=trace_both(image,left_lane_lines,right_lane_lines)
        #return img
        return diff_lane_color
        

    
    #cv2.imshow('image',diff_lane_color)'''
    
'''image=cv2.imread('./test_images/traffic1.jpg')
cv2.imshow('original',image)
cv2.imshow('image',display_final_lane_lines(image))
'''

video=cv2.VideoCapture('./test_videos/challenge.mp4')
#video=cv2.VideoCapture('./data/test photos and videos/road_traffic_2.mp4')

while(video.isOpened()):
    ret,image=video.read()
    if ret==True:
        cv2.imshow('output',display_final_lane_lines(image))
        if cv2.waitKey(1) == ord('q'):
            break
    else:
        break
video.release()
cv2.destroyAllWindows()
        
