from keras.layers import Dense, Activation, Conv2D, Flatten, MaxPooling2D
from keras.models import Sequential, load_model
from keras.optimizers import Adam
import numpy as np
import random
class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions, discrete=False):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.discrete = discrete
        self.state_memory = np.zeros((self.mem_size, input_shape[0],  input_shape[1],  input_shape[2]))
        self.new_state_memory = np.zeros((self.mem_size, input_shape[0],  input_shape[1],  input_shape[2]))
        dtype = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=dtype)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)
    
    def store_transition(self, current_state, action, reward, new_state, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] =current_state
        self.new_state_memory[index] = new_state
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - int(done)

        if self.discrete:
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1.0
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action

        self.mem_cntr +=1
    
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        current_states= self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return current_states, actions, rewards, new_states, terminal

def build_dqn(LR, n_actions, input_dims, fc1_dims, fc2_dims):
    print("hey")
    model = Sequential([
        Conv2D(32, kernel_size=(3,3),activation="relu"),
        MaxPooling2D(pool_size =(2, 2), strides =(2, 2)),
        Flatten(),
        Dense(fc1_dims),
        Activation("relu"),
        Dense(fc2_dims),
        Activation("relu"),
        Dense(n_actions)])
    
    model.compile(optimizer=Adam(learning_rate=LR), loss="mse")
    
    return model

class Agent(object):
    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size,
                 input_dims, epsilon_dec=0.996, epsilon_end=0.01, 
                 mem_size=1000000, fname='dqn_model.h5'):
        self.action_space = [0 for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions, discrete=False)
        self.q_eval = build_dqn(alpha, n_actions, input_dims, 256, 256)

    def remember(self, current_state, action, reward, new_state, done):
        self.memory.store_transition(current_state, action, reward, new_state, done)
    
    def choose_action(self, state):
        # print(state.shape)
        # exit()
        state = state[np.newaxis, :]
        rand = np.random.random()
        action = self.action_space.copy()
        # print(self.action_space)
        if rand < self.epsilon:
            action[0]=round(random.uniform(-0.7,0.7),2)
            action[1]=0.5
            action[2]=0
            # action = np.random.choice(self.action_space)
        else:
            action = self.q_eval.predict(state)[0]
            # action = np.argmax(actions)
        
        return action

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        # print(reward)
        action_values= np.array(self.action_space, dtype=np.int8)
        # action_indices = np.dot(action, action_values)

        q_eval = self.q_eval.predict(state)
        q_next = self.q_eval.predict(new_state)

        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        reward_new=[]
        done_new=[]
        for i in range(len(reward)):
            temp = [reward[i], reward[i], reward[i]]
            reward_new.append(temp)
            temp = [int(done[i]), int(done[i]), int(done[i])]
            done_new.append(temp)
        reward_new= np.array(reward_new)
        q_target[batch_index] = np.add(reward_new , self.gamma*np.multiply(q_next,done_new))
        _ = self.q_eval.fit(state, q_target, verbose = 0)
        self.epsilon = self.epsilon*self.epsilon_dec if self.epsilon > self.epsilon_min \
            else self.epsilon_min
    
    def save_model(self):
        self.q_eval.save(self.model_file)
    
    def load_model(self):
        self.q_eval = load_model(self.model_file)



