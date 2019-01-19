import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        
        self.hidden_size1 = 50
        self.hidden_size2 = 50
        
        self.fc1 = nn.Linear(state_size, self.hidden_size1)
        self.fc2 = nn.Linear(self.hidden_size1, self.hidden_size2)

        #dueling network
        self.fc3_adv = nn.Linear(self.hidden_size2, action_size)
        self.fc3_v = nn.Linear(self.hidden_size2, 1)


    def forward(self, state):
        """Build a network that maps state -> action values."""
        
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        adv = self.fc3_adv(x)
        val = self.fc3_v(x).expand(-1, adv.size(1))
        
        action = adv + val - adv.mean(1, keepdim=True).expand(-1, adv.size(1))
        
        return action
