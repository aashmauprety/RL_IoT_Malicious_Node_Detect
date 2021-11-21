# import required libraries
import numpy as np 
import pandas as pd
import random

# import env class from gym to create custom environment
from gym import Env  

# import two spaces from gym, allow us to define state and actions
from gym.spaces import Discrete, Box

# Declare required variables for the env
malicious_index = [0,1]
non_malicious_index = [2,3,4]

# Read the dataset and convert that to pandas Dataframe
def read_data():
        data = pd.read_csv("sample_data.csv")
        weight = pd.DataFrame(data)
        return weight

# Custom RL env to remove malicious node
class findmeEnv(Env):
    STATE_ELEMENTS = 5

    def __init__(self, weight):
        self.weight = weight
        self.action_space = Discrete(5)
        self.observation_space = Box(low=0, high=2, shape=(findmeEnv.STATE_ELEMENTS,)) #10 comm.round, 5 clients
        self.state = self.weight.iloc[0:1].values
        self.state1 = self.weight.iloc[0:1]
        # self.state = np.array([1.1,0.2,1.4,0.9,1.4], dtype=float)
      
        self.length = 100


    def step(self,action):
        if action==0:   #reward for this action is +2
            # self.state.iloc[:,random.choice(malicious_index)] = 0 #remove any one malicious weights(w1 or w2)
            a = random.choice(malicious_index)
            self.state1.iloc[:,a].replace([self.state1.iloc[:,a]],0,inplace=True)
            


        if action==1: #reward for this action is -10
            # self.state.iloc[:,random.choice(non_malicious_index)] = 0 #remove any one non-malicious node from w3,w4,w5
            b = random.choice(non_malicious_index)
            self.state1.iloc[:,b].replace([self.state1.iloc[:,b]],0,inplace=True)
            


        if action==2: #reward is +5
            # self.state.iloc[:,malicious_index] = 0 #remove both malicious nodes 
            self.state1.iloc[:,0].replace([self.state1.iloc[:,0]],0,inplace=True)
            self.state1.iloc[:,1].replace([self.state1.iloc[:,1]],0,inplace=True)
            


        if action==3: #reward is +1
            # self.state.iloc[:,[random.choice(malicious_index),random.choice(non_malicious_index)]] = 0 #remove one from malicious and one from non-malicious
            x = random.choice(malicious_index)
            y = random.choice(non_malicious_index)
            self.state1.iloc[:,x].replace([self.state1.iloc[:,x]],0,inplace=True)
            self.state1.iloc[:,y].replace([self.state1.iloc[:,y]],0,inplace=True)
            


        if action==4: #reward is -15
            # self.state.iloc[:,non_malicious_index] = 0 #remove all non-malicious nodes
            self.state1.iloc[:,2].replace([self.state1.iloc[:,2]],0,inplace=True)
            self.state1.iloc[:,3].replace([self.state1.iloc[:,3]],0,inplace=True)
            self.state1.iloc[:,4].replace([self.state1.iloc[:,4]],0,inplace=True)
            
        self.length -= 1

        if action==0:
            reward = 2
        if action==1:
            reward = -10
        if action == 2:
            reward = 5
        if action==3:
            reward = 1
        if action==4:
            reward = -15


        if self.length <= 0: 
            done = True
        else:
            done = False

        info = {}

        return self.state, reward, done, info



    def reset(self):
        self.state = self.weight.sample().reset_index(drop = True).values
        self.state = np.squeeze(self.state, axis=0)
        self.length = 100
        return self.state
        
