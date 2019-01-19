# **Project1. Navigation** 

## Report

---

[//]: # (Image References)
[image1]: ./result.png "Visualization"


**Learning Algorithm**
* double DQN\
 When predicting Q values for next states in the loss calculation, select action in latest Q network and calculate Q value in freezed Q network.\
 This implementation is shown in line 92, 93 of dqn_agent.py.

* dueling DQN\
 Neural network layer is splited advantage and value, finally added both descripted in model.py.

  My final model consisted of the following layers:

  | Layer         		    |   Description	           		                      |
  |:----------------------|:--------------------------------------------------| 
  | (1)Input         		  |   37 dimensions states                            | 
  | (2)Fully connected		|   input 37, outputs 50                            |
  | (3)ReLU		            |        									                          |
  | (4)Fully connected		|   input 50, outputs 50                            |
  | (5)ReLU		            |        									                          |
  | (6-1)Fully connected  |   input 50, outputs 4(advantage)                  |
  | (6-2)Fully connected  |   input 50, outputs 1(value)                      |
  | (7)Add		            |   advantage + value - mean(advantage) -> output 4 |


* hyperparameter\
 Used hyperparameters are shown below. 

  - BUFFER_SIZE = int(1e5)  # replay buffer size
  - BATCH_SIZE = 64         # minibatch size
  - GAMMA = 0.99            # discount factor
  - TAU = 1e-3              # for soft update of target parameters
  - LR = 5e-4               # learning rate 
  - UPDATE_EVERY = 4        # how often to update the network


**Plot of Rewards**\
 The scores of DQN with double DQN and dueling DQN show below.
 After 600 Episodes, average score exceeds 13.0. 

![alt text][image1]


A comparison between DQN and DQN(double DQN and dueling DQN) scores when they are learning is below.
Average window size is 100 episodes.\
THe results show better scores in DQN(double DQN and dueling DQN) than DQN.

* DQN\
  Episode 100	Average Score: 4.15\
  Episode 200	Average Score: 7.60\
  Episode 300	Average Score: 10.70\
  Episode 400	Average Score: 12.15\
  Episode 500	Average Score: 13.05\
  Episode 600	Average Score: 13.00\
  Episode 700	Average Score: 14.80\
  Episode 800	Average Score: 14.40\
  Episode 900	Average Score: 12.10\
  Episode 1000	Average Score: 14.70

* DQN(double DQN and dueling DQN)\
  Episode 100	Average Score: 1.20\
  Episode 200	Average Score: 4.65\
  Episode 300	Average Score: 9.45\
  Episode 400	Average Score: 12.10\
  Episode 500	Average Score: 12.80\
  Episode 600	Average Score: 16.05\
  Episode 700	Average Score: 15.40\
  Episode 800	Average Score: 16.15\
  Episode 900	Average Score: 16.90\
  Episode 1000	Average Score: 17.10

**Ideas for Future Work**\
  In Addition to current learning algorithm, prioritized replay buffer should be implemented.\
  I tried to implement it several times, but the calculation time was too long. So, it's not implemented this time.

