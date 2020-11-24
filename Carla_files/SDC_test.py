

#!/usr/bin/env python3


# CarlaUE4.exe /Game/Maps/RaceTrack -windowed -carla-server


# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Basic CARLA client example."""

from __future__ import print_function

import argparse
import logging
import random
import time
import cv2
from PIL import Image
import numpy as np
import os
from SDC_analysis import floodfill
import SDC_data_store
from carla import sensor
from carla import image_converter
import carla
from carla.client import make_carla_client
from carla.sensor import Camera, Lidar
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line
from SDC_pid import PID_throttle, PID_steer
import roughPPO
from SDC_state import state_update_long,state_update_lat
# import warnings
# warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

def ppo_input(measurements, trajectory, image_array, action_model, critic_model):
    player_measurements = measurements.player_measurements
    speed=player_measurements.forward_speed * 3.6
    # yaw = player_measurements.transform.rotation.yaw
    prev_throttle = SDC_data_store.prev_throttle
    prev_steer = SDC_data_store.turn_value
    state_array = np.array([speed, prev_throttle, prev_steer])
    if(len(trajectory))>20:
        trajectory = trajectory[:20]
    # action_output = action_model.predict([np.array([image_array]),
    #                                         np.array([state_array]), 
    #                                         np.array([trajectory])], 
    #                                         steps = 1)
    # critic_output = critic_model.predict([np.array([image_array]),
    #                                         np.array([state_array]), 
    #                                         np.array([trajectory]),
    #                                         action_output], 
    #                                         steps = 1)
    action_output = action_model.predict([tf.convert_to_tensor([image_array]),
                                          tf.convert_to_tensor([state_array]), 
                                          tf.convert_to_tensor([trajectory])], 
                                            steps = 1)
    critic_output = critic_model.predict([tf.convert_to_tensor([image_array]),
                                          tf.convert_to_tensor([state_array]), 
                                          tf.convert_to_tensor([trajectory]),
                                          tf.convert_to_tensor(action_output)], 
                                            steps = 1)
    # action_output_tf = tf.convert_to_tensor(action_output)
    # critic_output_tf = tf.convert_to_tensor(critic_output) 

    return action_output,critic_output


def run_carla_client(args):
    
    with make_carla_client(args.host, args.port) as client:
        print('CarlaClient connected')
        while True:

            settings = CarlaSettings()
            settings.set(
                SynchronousMode=True,
                SendNonPlayerAgentsInfo=True,
                NumberOfVehicles=0,
                NumberOfPedestrians=0,
                WeatherId=3, #1, 3, 7, 8, 14
                QualityLevel=args.quality_level)
            settings.randomize_seeds()

            # Now we want to add a couple of cameras to the player vehicle.
            # We will collect the images produced by these cameras every
            # frame.

            # The default camera captures RGB images of the scene.
            camera0 = sensor.Camera('CameraRGB',PostProcessing='SceneFinal')
            # Set image resolution in pixels.
            w=800
            h=600
            camera0.set_image_size(800, 600)
            # Set its position relative to the car in meters.
            camera0.set_position(0.30, 0, 4)
            settings.add_sensor(camera0)

            # Let's add another camera producing ground-truth depth.
            depth_camera = sensor.Camera('CameraDepth', PostProcessing='Depth')
            depth_camera.set_image_size(800, 600)
            depth_camera.set_position(x=0.30, y=0, z=4)
            settings.add_sensor(depth_camera)

            if args.lidar:
                lidar = Lidar('Lidar32')
                lidar.set_position(0, 0, 2.50)
                lidar.set_rotation(0, 0, 0)
                lidar.set(
                    Channels=32,
                    Range=50,
                    PointsPerSecond=100000,
                    RotationFrequency=10,
                    UpperFovLimit=10,
                    LowerFovLimit=-30)
                settings.add_sensor(lidar)

            scene = client.load_settings(settings)

            # Choose one player start at random.
            number_of_player_starts = len(scene.player_start_spots)
            player_start = random.randint(0, max(0, number_of_player_starts - 1))

            print('Starting new episode at %r...' % scene.map_name)
            client.start_episode(player_start)
            action_model = roughPPO.actor_model((h,w,3),(3,1),(20,2))
            critic_model = roughPPO.critic_model((h,w,3),(3,1),(20,2),(2,1))
            actor_action = [0,0]
            critic_score = 0
            ppo_check = 0
            # Iterate every frame in the episode.
            ppo_iter_count = 0
            while True:

                    
                # Read the data produced by the server this frame.
                
                measurements, sensor_data = client.read_data()
                # print_measurements(measurements)
                main_image=sensor_data.get("CameraRGB",None)

                image_array=image_converter.to_rgb_array(main_image)
                
                point_cloud = image_converter.depth_to_local_point_cloud(
                    sensor_data['CameraDepth'],
                    image_array,
                    max_depth=1000
                )

                image_array=cv2.cvtColor(image_array,cv2.COLOR_RGB2BGR)
                cv2.imshow("FrontCam View",image_array)
                reverse = 0
                try:                    
                    steer_val = 0
                    
                    bez_mask, coord_trajectory, trajectory,world_x_left,world_x_right,world_z_left,world_z_right=floodfill(image_array, point_cloud)
                    
                    track_point_yaw = trajectory[-5]
                    yaw_error = np.arctan((0 - track_point_yaw[0])/(track_point_yaw[1]))
                    state_update_long(trajectory)
                    state_update_lat(world_x_left, world_x_right, yaw_error)

                    if ppo_check ==1:
                        reward_score = roughPPO.reward(actor_action, world_x_left, world_x_right)
                        if len(SDC_data_store.throttle_queue) < 50:
                            SDC_data_store.throttle_queue.append(actor_action[0])
                            SDC_data_store.steer_queue.append(actor_action[1])
                            SDC_data_store.rewards_queue.append(reward_score)
                            SDC_data_store.critic_score.append(critic_score)
                        else:
                            print("IDHAR!!!!!!!!!!")
                            SDC_data_store.throttle_queue.popleft()
                            SDC_data_store.throttle_queue.append(actor_action[0])
                            SDC_data_store.steer_queue.popleft()
                            SDC_data_store.steer_queue.append(actor_action[1])
                            SDC_data_store.rewards_queue.popleft()
                            SDC_data_store.rewards_queue.append(reward_score)
                            SDC_data_store.critic_score.popleft()                        
                            SDC_data_store.critic_score.append(reward_score)


                        if(len(SDC_data_store.rewards_queue)>1):
                            returns , advs = roughPPO.advantages(list(SDC_data_store.critic_score), 
                                                                list(SDC_data_store.rewards_queue))
                            loss = roughPPO.ppo_loss(advs,SDC_data_store.rewards_queue,SDC_data_store.critic_score)
                            action_model.compile(optimizer=Adam(lr = 1e-3), loss = loss)
                            critic_model.compile(optimizer=Adam(lr = 1e-3), loss = 'mse')
                            print("LOSS : "+str(loss))
                        ppo_check = 0
                    if ppo_iter_count%20 == 0 and ppo_iter_count>0 and ppo_check ==0:
                        ppo_check = 1                            
                        action_output, critic_output = ppo_input(measurements, trajectory, image_array, action_model, critic_model)
                        actor_action = action_output[0]
                        critic_score = critic_output[0][0]
                        throttle_a = actor_action[0]
                        steer_val = actor_action[1]

                    else:
                        throttle_a = PID_throttle(trajectory, reverse)                   
                        cv2.imshow("bez",bez_mask)
                        steer_val = PID_steer(trajectory, reverse, world_x_left,world_x_right)
                    
                    check=SDC_data_store.shadow_check

                    client.send_control(
                        steer=steer_val*check,
                        throttle=throttle_a*check + 0.5*(1-check),
                        brake=0.0,
                        hand_brake=False,
                        reverse=False)
                    SDC_data_store.turn_value=steer_val
                except Exception as e:
                    print(e)

                    client.send_control(
                        steer=SDC_data_store.turn_value*check,
                        throttle=1,
                        brake=0.85,
                        hand_brake=True,
                        reverse=False)         
                
                cv2.waitKey(1)

                if SDC_data_store.count%10==0:
                        print("frame: ",SDC_data_store.count)
                
                # if SDC_data_store.count%10==0:
                #     SDC_data_store.sum_cte = 0
                SDC_data_store.count +=1
                print("*********\n")
                ppo_iter_count+=1

                

                

def print_measurements(measurements):
    number_of_agents = len(measurements.non_player_agents)
    player_measurements = measurements.player_measurements
    message = 'Vehicle at ({pos_x:.1f}, {pos_y:.1f}), '
    message += '{speed:.0f} km/h, '
    message += 'Collision: {{vehicles={col_cars:.0f}, pedestrians={col_ped:.0f}, other={col_other:.0f}}}, '
    message += '{other_lane:.0f}% other lane, {offroad:.0f}% off-road, '
    message += '{yaw:.0f} Yaw '
    message += '({agents_num:d} non-player agents in the scene)'
    message = message.format(
        pos_x=player_measurements.transform.location.x,
        pos_y=player_measurements.transform.location.y,
        speed=player_measurements.forward_speed * 3.6, # m/s -> km/h
        col_cars=player_measurements.collision_vehicles,
        col_ped=player_measurements.collision_pedestrians,
        col_other=player_measurements.collision_other,
        other_lane=100 * player_measurements.intersection_otherlane,
        offroad=100 * player_measurements.intersection_offroad,
        yaw = player_measurements.transform.rotation.yaw,
        agents_num=number_of_agents)

    print_over_same_line(message)


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '-l', '--lidar',
        action='store_true',
        help='enable Lidar')
    argparser.add_argument(
        '-q', '--quality-level',
        choices=['Low', 'Epic'],
        type=lambda s: s.title(),
        default='Epic',
        help='graphics quality level, a lower level makes the simulation run considerably faster.')
    argparser.add_argument(
        '-i', '--images-to-disk',
        action='store_true',
        dest='save_images_to_disk',
        help='save images (and Lidar data if active) to disk')
    argparser.add_argument(
        '-c', '--carla-settings',
        metavar='PATH',
        dest='settings_filepath',
        default=None,
        help='Path to a "CarlaSettings.ini" file')

    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    args.out_filename_format = '_out/episode_{:0>4d}/{:s}/{:0>6d}'
    SDC_data_store.prev_data_init()
    # SDC_data_store.neural_net_data()
    while True:
        try:

            run_carla_client(args)

            print('Done.')
            return

        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
