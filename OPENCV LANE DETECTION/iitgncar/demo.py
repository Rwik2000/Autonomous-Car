import os
import time
import sys
# from docopt import docopt
import numpy as np
import cv2
from simple_pid import PID
import gym
import gym_donkeycar
import donkeycar as dk
# from donkeycar.parts.datastore import TubHandler
from SDC.turn_detection_main import grabCut
from SDC.turn_detection import floodfill
class LineFollower:
    '''
    OpenCV based controller
    This controller takes a horizontal slice of the image at a set Y coordinate.
    Then it converts to HSV and does a color thresh hold to find the yellow pixels.
    It does a histogram to find the pixel of maximum yellow. Then is uses that iPxel
    to guid a PID controller which seeks to maintain the max yellow at the same point
    in the image.
    '''
    def __init__(self):
        self.vert_scan_y = 60   # num pixels from the top to start horiz scan
        self.vert_scan_height = 10 # num pixels high to grab from horiz scan
        self.color_thr_low = np.asarray((0, 50, 50)) # hsv dark yellow
        self.color_thr_hi = np.asarray((50, 255, 255)) # hsv light yellow
        self.target_pixel = None # of the N slots above, which is the ideal relationship target
        self.steering = 0.0 # from -1 to 1
        self.throttle = 0.0 # from -1 to 1
        self.recording = False # Set to true if desired to save camera frames
        self.delta_th = 0.1 # how much to change throttle when off
        self.throttle_max = 0.3
        self.throttle_min = 0.15
        self.pid_st = PID(Kp=-0.01, Ki=0.00, Kd=-0.001)


    def run(self, cam_img):
        '''
        main runloop of the CV controller
        input: cam_image, an RGB numpy array
        output: steering, throttle, and recording flag
        '''

        # cam_img=cv2.resize(cam_img,(800,600))
        # print(cam_img.shape)
        # turn_value=floodfill(cam_img)
        # self.throttle=0.05
        # self.steering=turn_value*2
        turn_value=floodfill(cam_img)
        self.throttle=0.05
        self.steering=turn_value*2
        # try:
            
        #     # turn_value=grabCut(img=cam_img,prevImg=cam_img)
            
        # except:
        #     pass
        # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cam_img=cv2.resize(cam_img,(800,600))
        cv2.imshow("image", cam_img)
        # cv2.resizeWindow('image', 800,600)
        # print("steering:",self.steering)
        cv2.waitKey(1)
        return self.steering, self.throttle, self.recording


def drive(cfg, args):
    '''
    Construct a working robotic vehicle from many parts.
    Each part runs as a job in the Vehicle loop, calling either
    it's run or run_threaded method depending on the constructor flag `threaded`.
    All parts are updated one after another at the framerate given in
    cfg.DRIVE_LOOP_HZ assuming each part finishes processing in a timely manner.
    Parts may have named outputs and inputs. The framework handles passing named outputs
    to parts requesting the same named input.
    '''
    
    #Initialize car
    V = dk.vehicle.Vehicle()

    #Camera
    if cfg.DONKEY_GYM:
        from donkeycar.parts.dgym import DonkeyGymEnv
        cfg.GYM_CONF['racer_name'] = args[1]
        cfg.GYM_CONF['car_name'] = args[1]
        cam = DonkeyGymEnv(cfg.DONKEY_SIM_PATH, host=cfg.SIM_HOST, env_name=cfg.DONKEY_GYM_ENV_NAME, conf=cfg.GYM_CONF, delay=cfg.SIM_ARTIFICIAL_LATENCY)
        inputs = ['steering', 'throttle']
    else:
        from donkeycar.parts.camera import PiCamera
        cam = PiCamera(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH)
        inputs = []

    V.add(cam, inputs=inputs, outputs=['cam/image_array'], threaded=True)
        
    #Controller
    V.add(LineFollower(), 
          inputs=['cam/image_array'],
          outputs=['steering', 'throttle', 'recording'])

        
    #Drive train setup
    # if not cfg.DONKEY_GYM:
    #     from donkeycar.parts.actuator import PCA9685, PWMSteering, PWMThrottle

    #     steering_controller = PCA9685(cfg.STEERING_CHANNEL, cfg.PCA9685_I2C_ADDR, busnum=cfg.PCA9685_I2C_BUSNUM)
    #     steering = PWMSteering(controller=steering_controller,
    #                                     left_pulse=cfg.STEERING_LEFT_PWM, 
    #                                     right_pulse=cfg.STEERING_RIGHT_PWM)
        
    #     throttle_controller = PCA9685(cfg.THROTTLE_CHANNEL, cfg.PCA9685_I2C_ADDR, busnum=cfg.PCA9685_I2C_BUSNUM)
    #     throttle = PWMThrottle(controller=throttle_controller,
    #                                     max_pulse=cfg.THROTTLE_FORWARD_PWM,
    #                                     zero_pulse=cfg.THROTTLE_STOPPED_PWM, 
    #                                     min_pulse=cfg.THROTTLE_REVERSE_PWM)

    #     V.add(steering, inputs=['steering'])
    #     V.add(throttle, inputs=['throttle'])
    
    # #add tub to save data

    # inputs=['cam/image_array',
    #         'steering', 'throttle']

    # types=['image_array',
    #         'float', 'float']

    # th = TubHandler(path=cfg.DATA_PATH)
    # tub = th.new_tub_writer(inputs=inputs, types=types)
    # V.add(tub, inputs=inputs, outputs=["tub/num_records"], run_condition="recording")

    #run the vehicle
    V.start(rate_hz=cfg.DRIVE_LOOP_HZ, 
            max_loop_count=cfg.MAX_LOOPS)


if __name__ == '__main__':
    # args = docopt(__doc__)
    args=sys.argv
    cfg = dk.load_config()
    drive(cfg, args)
    