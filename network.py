import torch
import torch.nn as nn
import torch.optim as optim

class Network(nn.Module):
    
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(Network, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        
        self.first_net = nn.Sequential(
            nn.Linear(self.input_dims, self.fc1_dims),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(self.fc1_dims, self.fc2_dims),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(self.fc2_dims, self.n_actions[0])
            )
        
    
        self.second_net = nn.Sequential(
            nn.Linear(self.input_dims, self.fc1_dims),
            nn.Dropout(0.6),
            nn.ReLU(),
            nn.Linear(self.fc1_dims, self.fc2_dims),
            nn.Dropout(0.6),
            nn.ReLU(),
            nn.Linear(self.fc2_dims, self.n_actions[1])
            )
        
        
        self.third_net = nn.Sequential(
            nn.Linear(self.input_dims, self.fc1_dims),
            nn.Dropout(0.6),
            nn.ReLU(),
            nn.Linear(self.fc1_dims, self.fc2_dims),
            nn.Dropout(0.6),
            nn.ReLU(),
            nn.Linear(self.fc2_dims, self.n_actions[2])
            )
        
        self.optimizer = optim.RMSprop(self.parameters(), lr = lr)
        self.device = torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')
        
    def get_action(self, x):
        first_action = torch.argmax(self.first_net(x), 1).unsqueeze(1)
        second_action = torch.argmax(self.second_net(x), 1).unsqueeze(1)
        third_action = torch.argmax(self.third_net(x), 1).unsqueeze(1)
        return torch.cat((first_action, second_action, third_action), 1)
        
    def forward(self, x):
        first_state_value = self.first_net(x)
        second_state_value = self.second_net(x)
        third_state_value = self.third_net(x)
        return first_state_value, second_state_value, third_state_value