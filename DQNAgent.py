import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from memory import * 
from network import *
from tqdm import tqdm

class DQNAgent():
    
    def __init__(self, env, gamma, epsilon, lr, input_dims, batch_size, n_actions = 3, 
                 max_mem_size = 100000, eps_min = 0.01, eps_dec = 5e-4):
        
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.lr = lr
        self.max_mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.device = torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')
        self.memory = ReplayMemory(max_mem_size)
        self.criterion = nn.SmoothL1Loss()
        self.losses = []
        self.avg_losses = []
        self.TARGET_UPDATE = 10
        
        self.policy_net = Network(lr, input_dims, fc1_dims = 256, fc2_dims = 256, n_actions = env.action_space.high)
        self.target_net = Network(lr, input_dims, fc1_dims = 256, fc2_dims = 256, n_actions = env.action_space.high)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        #for param in self.target_net.parameters():
        #    param.requires_grad = False
        
    def add_memory(self, state, action, next_state, reward):
        self.memory.push(state, action, next_state, reward)
        
    def select_action(self):
        if random.random() > self.epsilon:
            state = torch.tensor(self.env.state).unsqueeze(0)
            with torch.no_grad():
                state = torch.tensor(state).float()
                #state.to(self.device)
                action = self.policy_net.get_action(state)
        else:
            action = [random.randint(0,99), random.randint(0,89), random.randint(0,79)]
            action = torch.tensor([action])
        return action
    
    def update_epsilon(self):
        if self.epsilon > self.eps_min:
            self.epsilon = self.epsilon - self.eps_dec
        else:
            self.epsilon = self.eps_min
    
    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        self.policy_net.optimizer.zero_grad()
        max_mem = min(self.mem_cntr, self.max_mem_size)
        
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=self.device, dtype=torch.bool).reshape(self.batch_size, -1)
        
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None]).reshape(self.batch_size, -1).float()

        state_batch = torch.cat(batch.state).reshape(self.batch_size, -1).float()
        action_batch = torch.cat(batch.action).reshape(self.batch_size, -1).type(torch.int64)
        reward_batch = torch.cat(batch.reward).reshape(self.batch_size, -1)
        
        first_stage_values, second_stage_values, third_stage_values = self.policy_net(state_batch)
        
        first_state_action_values = first_stage_values.gather(1, action_batch[:,0].unsqueeze(1))
        second_state_action_values = second_stage_values.gather(1, action_batch[:,1].unsqueeze(1))
        third_state_action_values = third_stage_values.gather(1, action_batch[:,2].unsqueeze(1))
        
        state_action_values = torch.cat((first_state_action_values, second_state_action_values, third_state_action_values), 1)
        
        #with torch.no_grad():
        first_next_state_values, second_next_state_values, third_next_state_values = self.target_net(non_final_next_states)
        
        first_next_state_values = first_next_state_values.max(1)[0].detach().unsqueeze(1)
        second_next_state_values = first_next_state_values.max(1)[0].detach().unsqueeze(1)
        third_next_state_values = first_next_state_values.max(1)[0].detach().unsqueeze(1)
        
        next_state_values = torch.cat((first_next_state_values, second_next_state_values, third_next_state_values), 1)
        
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch.float()
        
        loss = self.criterion(state_action_values, expected_state_action_values)
        
        loss.backward()
        self.losses.append(loss.item())
        self.policy_net.optimizer.step()
        
    def optimize_model(self, num_episodes):
        pbar = tqdm(total=num_episodes)
        self.env.reset()
        state = torch.tensor(self.env.state).unsqueeze(0)
        episode_count = 0
        done = False
        while episode_count < num_episodes:
            action = self.select_action().numpy()[0]
            if done:
                self.env.reset()
                self.avg_losses.append(np.mean(self.losses))
                self.losses = []
                episode_count += 1
                pbar.update(1)
            next_state, reward, done, _ = self.env.step(action)
            state = torch.tensor(state)
            next_state = torch.tensor(next_state).unsqueeze(0)
            reward = torch.tensor([reward])
            action = torch.tensor(action)
            
            self.memory.push(state, action, next_state, reward)
            
            state = next_state
            self.train()
            self.update_epsilon()
            if episode_count  % self.TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
        pbar.close()