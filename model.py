from Rl_Env import findmeEnv
from Rl_Env import read_data
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import random 
import time


weight = read_data()
env = findmeEnv(weight)

# Get shapes for observation space and action space of the env
def get_shape():
    states = env.observation_space.shape 
    actions = env.action_space.n 
    return states, actions

# Function for defining model architecture 
def build_model(states,actions):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape = (1,5))) 
    model.add(Dense(64, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    model.add(Flatten())
    return model