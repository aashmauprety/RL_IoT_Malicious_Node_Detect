from Rl_Env import findmeEnv
from Rl_Env import read_data
from model import build_model, get_shape
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

weight = read_data()
env = findmeEnv(weight)
states,actions = get_shape()

#build the model as input and output shape
model = build_model(states,actions)
print(model.summary)

# Define RL agent 
def build_agent(model, actions):
  policy = BoltzmannQPolicy()
  memory = SequentialMemory(limit=50000, window_length = 1)
  dqn = DQNAgent(model = model, memory = memory, policy=policy, nb_actions=actions, nb_steps_warmup = 10, target_model_update = 1e-2)
  return dqn


dqn = build_agent(model, actions)
dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])
dqn.fit(env, nb_steps=50, visualize=False,verbose=1)
