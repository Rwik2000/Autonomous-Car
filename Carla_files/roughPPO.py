import numpy as np
import tensorflow as tf 
import gym
import tensorflow_probability as tfp
import tensorflow.keras.losses as kls
import SDC_data_store 
from tensorflow.keras.layers import Dense, Flatten, concatenate, Conv2D
from tensorflow.keras import Model, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2


def actor_model(image_dims, state_dims, traj_dims):
    image_input = Input(shape=image_dims)
    car_state_input = Input(shape=state_dims)
    trajectory_input = Input(shape=traj_dims)

    image_NN = Conv2D(128,(3,3))(image_input)
    image_NN = Dense(32,activation='relu')(image_NN)
    image_NN = Flatten()(image_NN)
    image_NN = Dense(4, activation='relu')(image_NN)
    
    image_NN = Model(inputs = image_input,outputs = image_NN)

    state_NN = Dense(512, activation='relu')(car_state_input)
    state_NN = Flatten()(state_NN)
    state_NN = Dense(4, activation= 'relu')(state_NN)
    state_NN = Model(inputs = car_state_input, outputs = state_NN)

    traj_NN = Dense(32, activation='relu')(trajectory_input)
    traj_NN=Flatten()(traj_NN)
    traj_NN = Dense(4,activation='relu')(traj_NN)
    traj_NN = Model(inputs = trajectory_input, outputs = traj_NN)

    combined_NN = concatenate([image_NN.output, state_NN.output, traj_NN.output])
    combined_NN = Dense(64, activation='relu')(combined_NN)
    combined_NN = Dense(2)(combined_NN)

    model = Model(inputs = [image_NN.input, state_NN.input, traj_NN.input], outputs = combined_NN)
    model.compile(optimizer=Adam(lr=1e-4), loss='mse')
    return model

def critic_model(image_dims, state_dims,traj_dims,action_dims):
    image_input = Input(shape=image_dims)
    car_state_input = Input(shape=state_dims)
    action_input = Input(shape=action_dims)
    trajectory_input = Input(shape=traj_dims)

    image_NN = Conv2D(128,(3,3))(image_input)
    image_NN = Dense(32,activation='relu')(image_NN)
    image_NN = Flatten()(image_NN)
    image_NN = Dense(4, activation='relu')(image_NN)
    
    image_NN = Model(inputs = image_input,outputs = image_NN)

    state_NN = Dense(512, activation='relu')(car_state_input)
    state_NN = Flatten()(state_NN)
    state_NN = Dense(4, activation= 'relu')(state_NN)
    state_NN = Model(inputs = car_state_input, outputs = state_NN)

    action_NN = Dense(32, activation='relu')(action_input)
    action_NN = Flatten()(action_NN)
    action_NN = Dense(4,activation='relu')(action_NN)
    action_NN = Model(inputs = action_input, outputs = action_NN)

    traj_NN = Dense(32, activation='relu')(trajectory_input)
    traj_NN=Flatten()(traj_NN)
    traj_NN = Dense(4,activation='relu')(traj_NN)
    traj_NN = Model(inputs = trajectory_input, outputs = traj_NN)

    combined_NN = concatenate([image_NN.output, state_NN.output, traj_NN.output,action_NN.output ])
    combined_NN = Dense(64, activation='relu')(combined_NN)
    combined_NN = Dense(1)(combined_NN)

    model = Model(inputs = [image_NN.input, state_NN.input, traj_NN.input ,action_NN.input], outputs = combined_NN)
    return model

def advantages(critic_values, rewards):
    returns = []
    gae = 0
    gamma = 0.99
    lmbda = 0.95
    for i in reversed(range(len(rewards)-1)):
        delta = rewards[i] + gamma * critic_values[i + 1] - critic_values[i]
        gae = delta + gamma * lmbda * gae
        returns.insert(0, gae + critic_values[i])

    adv = np.array(returns) - critic_values[:-1]
    # print(adv)
    return returns, (adv - np.mean(adv)) / (np.std(adv) + 1e-10)

def ppo_loss(advantages, rewards, values):
    rewards = np.array(list(rewards))
    values = np.array(list(values))
    new_action = np.array([SDC_data_store.throttle_queue[-1], SDC_data_store.steer_queue[-1]])
    try:
        old_action = np.array([list(SDC_data_store.throttle_queue)[:-1], list(SDC_data_store.steer_queue)[:-1]])
        old_action = np.transpose(old_action)
    except:
        old_action = np.array([SDC_data_store.throttle_queue, SDC_data_store.steer_queue])
    mean_list = np.array([np.mean(SDC_data_store.throttle_queue), np.mean(SDC_data_store.steer_queue)])
    std_dev_list = np.array([np.std(SDC_data_store.throttle_queue), np.std(SDC_data_store.steer_queue)])
    newpolicy_probs = 1/(np.sqrt(2*np.pi)*std_dev_list)*np.exp(-(new_action - mean_list)/(2*std_dev_list**2))
    oldpolicy_probs = 1/(np.sqrt(2*np.pi)*std_dev_list)*np.exp(-(old_action - mean_list)/(2*std_dev_list**2))
    ratio = np.exp(np.log(newpolicy_probs + 1e-10) - np.log(oldpolicy_probs + 1e-10))
    ratio = np.transpose(ratio)
    clipping_val = 0.2
    critic_discount = 0.5
    entropy_beta = np.log(np.sqrt(2*np.pi*np.exp(1)*std_dev_list**2))

    p1 = ratio * advantages
    p2 = np.clip(ratio, 1 - clipping_val, 1 + clipping_val) * advantages[0]
    actor_loss = -np.mean(np.minimum(p1, p2))
    critic_loss = np.mean(np.square(rewards - values))
    total_loss = critic_discount * critic_loss + actor_loss 
    # - entropy_beta * np.mean(
    #                                     -(newpolicy_probs * np.log(newpolicy_probs + 1e-10)))
    return total_loss

def reward(action, x_left,x_right):
    # print(action)
    left_tolerance_queue = SDC_data_store.left_tolerance_queue
    right_tolerance_queue = SDC_data_store.right_tolerance_queue
    cte_queue = SDC_data_store.cte_queue
    yaw_error_queue = SDC_data_store.yaw_error_queue

    prev_yaw_error = SDC_data_store.prev_yaw_error
    prev_throttle = SDC_data_store.prev_throttle
    prev_turn_val =SDC_data_store.prev_turn_val
    prev_cte = SDC_data_store.prev_cte

    throttle_score = -0.05*abs(action[0] - prev_throttle)
    yaw_score = -0.01*abs(prev_yaw_error)
    steer_score = -0.01*abs(action[1] - prev_turn_val)

    avg_yaw_error_score = -0.01*(sum(list(yaw_error_queue)))
    avg_side_tol_score = -0.01*(sum(list(left_tolerance_queue))
                                 + sum(list(right_tolerance_queue)))
    side_tolerance_score = -0.01*(round(np.exp(-(x_left[0] - 0)),5)
                                 +round(np.exp((x_right[0] - 0)),5))
    cte_score = -0.01*abs(prev_cte) - 0.005*sum(list(cte_queue))

    throttle_tolerance_score = 0
    steer_tolerance_score = 0
    if(abs(action[0])>1):
        throttle_tolerance_score = -10
    elif(abs(action[0])<0.6):
        throttle_tolerance_score = -2

    if abs(action[1]>2.5):
        steer_tolerance_score = -10
    
    total_score =(throttle_score + 
                  yaw_score + 
                  steer_score + 
                  avg_yaw_error_score + 
                  avg_side_tol_score + 
                  side_tolerance_score + 
                  cte_score + 
                  throttle_tolerance_score +
                  steer_tolerance_score)
    
    return total_score

# x_left after update
# x_right after update
# throttle before update
# steer before update
# yaw_after update



         