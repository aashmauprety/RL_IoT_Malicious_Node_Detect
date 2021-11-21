from Rl_Env import findmeEnv
from Rl_Env import read_data
from model import build_model, get_shape
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import ModelIntervalCheckpoint, FileLogger

weight = read_data()
env = findmeEnv(weight)
states,actions = get_shape()

#build the model as input and output shape
model = build_model(states,actions)
print(model.summary)

# Define call backs
ENV_NAME = 'RL_IoT-v1'
def build_callbacks(ENV_NAME):
    checkpoint_weights_filename = 'dqn_' + ENV_NAME + '_weights_{step}.h5f'
    log_filename = 'dqn_{}_log.json'.format(ENV_NAME)
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
    callbacks += [FileLogger(log_filename, interval=100)]
    return callbacks


# Define RL agent 
policy = BoltzmannQPolicy()
memory = SequentialMemory(limit=50000, window_length = 1)
dqn = DQNAgent(model=model, nb_actions=actions, policy=policy, memory=memory, nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
                   train_interval=4, delta_clip=1.)
dqn.compile(Adam(lr=.00025), metrics=['mae'])
callbacks = build_callbacks(ENV_NAME)

#Start Training
dqn.fit(env, nb_steps=20000, visualize=False,verbose=2, callbacks=callbacks)
dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)


