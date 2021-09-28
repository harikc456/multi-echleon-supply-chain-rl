import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from memory import * 
from network import *
from tqdm import tqdm

class DDQNAgent():
    
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
        
        self.q1_net = Network(lr, input_dims, fc1_dims = 256, fc2_dims = 256, n_actions = env.action_space.high)
        self.q2_net = Network(lr, input_dims, fc1_dims = 256, fc2_dims = 256, n_actions = env.action_space.high)
        self.q2_net.load_state_dict(self.q1_net.state_dict())
        
        self.action_select_net = self.q1_net
        self.action_eval_net = self.q2_net
        
    def add_memory(self, state, action, next_state, reward):
        self.memory.push(state, action, next_state, reward)
        
    def select_action(self, state = torch.tensor([])):
        if random.random() > self.epsilon:
            if len(state) == 0:
                state = torch.tensor(self.env.state).unsqueeze(0)
            with torch.no_grad():
                state = torch.tensor(state).float()
                action = self.action_select_net.get_action(state)
        else:
            if state.dim() > 1:
                action = []
                for _ in range(state.size()[0]):
                    random_action = [random.randint(0,99), random.randint(0,89), random.randint(0,79)]
                    action.append(random_action)
            else:
                action = [random.randint(0,99), random.randint(0,89), random.randint(0,79)]
        return np.array(action)
    
    def update_epsilon(self):
        if self.epsilon > self.eps_min:
            self.epsilon = self.epsilon - self.eps_dec
        else:
            self.epsilon = self.eps_min
    
    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        if np.random.random(1)[0] < 0.5:
            self.action_select_net = self.q1_net
            self.action_eval_net = self.q2_net
        else:
            self.action_select_net = self.q2_net
            self.action_eval_net = self.q1_net
            
        self.action_select_net.optimizer.zero_grad()
        
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=self.device, dtype=torch.bool).reshape(self.batch_size, -1)
        
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None]).reshape(self.batch_size, -1).float()

        state_batch = torch.cat(batch.state).reshape(self.batch_size, -1).float()
        action_batch = torch.cat(batch.action).reshape(self.batch_size, -1).type(torch.int64)
        reward_batch = torch.cat(batch.reward).reshape(self.batch_size, -1)
        
        pred_first_values, pred_second_values, pred_third_values = self.action_select_net(state_batch)
        
        first_state_values = pred_first_values.gather(1, action_batch[:,0].unsqueeze(1))
        second_state_values = pred_second_values.gather(1, action_batch[:,1].unsqueeze(1))
        third_state_values = pred_third_values.gather(1, action_batch[:,2].unsqueeze(1))
        
        pred_state_action_values = torch.cat((first_state_values, second_state_values, third_state_values), 1)
       
        pred_next_value_indices = torch.tensor(self.select_action(non_final_next_states)).type(torch.int64)
        
        target_first_values, target_second_values, target_third_values = self.action_eval_net(non_final_next_states)
        
        target_first_values = target_first_values.gather(1, pred_next_value_indices[:,0].detach().unsqueeze(1))
        target_second_values = target_second_values.gather(1, pred_next_value_indices[:,1].detach().unsqueeze(1))
        target_third_values = target_third_values.gather(1, pred_next_value_indices[:,2].detach().unsqueeze(1))
        
        target_state_action_values = torch.cat((target_first_values, target_second_values, target_third_values), 1)
        
        expected_state_action_values = (target_state_action_values * self.gamma) + reward_batch.float()
            
        loss = self.criterion(pred_state_action_values, expected_state_action_values)
        loss.backward()
        self.losses.append(loss.item())
        self.action_select_net.optimizer.step()
        
    def optimize_model(self, num_episodes):
        pbar = tqdm(total=num_episodes)
        self.env.reset()
        state = self.env.state
        episode_count = 0
        done = False
        while episode_count < num_episodes:
            action = self.select_action()
            if len(action.shape) == 2:
                action = action[0]
            if done:
                self.env.reset()
                self.avg_losses.append(np.mean(self.losses))
                self.losses = []
                episode_count += 1
                pbar.update(1)
        
            next_state, reward, done, _ = self.env.step(action)
            state = torch.tensor(state)
            next_state = torch.tensor(next_state)
            reward = torch.tensor([reward])
            action = torch.tensor(action)
            
            self.memory.push(state, action, next_state, reward)
            
            state = next_state
            self.train()
            self.update_epsilon()
        pbar.close()