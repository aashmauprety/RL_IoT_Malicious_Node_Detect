> This repo includes code to research paper titled **"Detection of malicious agents in Federated Learning Empowered IoT Nodes Using Reinforcement Learning"**
Here an custom RL env is written in Python and agent is trained on the environment using DQN algorithm. 
Please refer to the paper for detail on environment, state space, action space and reward. 

Following is the steps to follow to run this project:
+ Install all the required libraries using ``` $ pip install -r requirements.txt```
+ Run the ```main.py``` using ```python main.py``` command. This will start training of the agent
+ Run ```test.py``` to use the trained RL model for prediction, which is to identify malicious IoT nodes using simulated sample data.
