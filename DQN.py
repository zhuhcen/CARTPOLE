from typing import Any, Mapping
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from ReplayBuffer import ReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FCN(nn.Module):
    def __init__(self, alpha, state_dim, action_dim, hidden_dim1=256, hidden_dim2=256) -> None:
        '''
        初始化全链接网络，alpha为学习率，action_dim为动作空间维度，state_dim为状态空间（观测空间）维度
        hidden_dim1为第一层隐藏层维度，hidden_dim2为第二层隐藏层维度。
        '''
        super().__init__()
        self.alpha = alpha
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2

        self.fc1 = nn.Linear(self.state_dim, self.hidden_dim1)
        self.fc2 = nn.Linear(self.hidden_dim1, self.hidden_dim2)
        self.q = nn.Linear(self.hidden_dim2, self.action_dim)

        self.optimizer = optim.Adam(self.parameters(), self.alpha)
        self.to(device)

    def forward(self, state : torch.Tensor) -> torch.Tensor:
        fc1_out_activated = torch.relu(self.fc1(state))
        fc2_out_activated = torch.relu(self.fc2(fc1_out_activated))
        q_out = self.q(fc2_out_activated)
        return q_out
    
    def save_model(self, ckpt_dir):
        torch.save(self.state_dict, ckpt_dir)
    
    def load_model(self, ckpt_dir):
        self.load_state_dict(torch.load(ckpt_dir))
    
'''
DQN要素
--------
-决策过程
--能根据当前状态选择动作（choose_action）
-数据收集
--将探索过程中的经验收集到replay_buffer中（remember）
-训练过程
--根据replay_buffer中的数据，以及公式得出损失函数，进行梯度更新(learn)
--评估网络与目标网络 参数之间的更新(update_network_parameters)
'''
class DQN:
    def __init__(self, alpha: float, state_dim: int, action_dim: int, hidden_dim1: int, hidden_dim2: int, chpt_dir: str, # 全连接网络相关
                 tau:float, gamma:float, epsilon: float, eps_min: float, eps_dec: float, # DQN算法相关
                 max_size: int, batch_size: int) -> None:
        self.alpha = alpha
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.ckpt_dir = chpt_dir
        self.tau = tau
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.max_size = max_size
        self.batch_size = batch_size

        self.eval_nn = FCN(self.alpha, self.state_dim, self.action_dim, self.hidden_dim1, self.hidden_dim2)
        self.target_nn = FCN(self.alpha, self.state_dim, self.action_dim, self.hidden_dim1, self.hidden_dim2)

        self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim, self.max_size, self.batch_size)

        self.update_network_parameters(tau=1)


    def update_network_parameters(self, tau = None):
        '''
        tau等于1时为hard update，表示target网络完全使用eval网络的参数'''
        if tau == None:
            tau = self.tau
        
        for q_eual_param, q_target_param in zip(self.eval_nn.parameters(), self.target_nn.parameters()):
            q_target_param.data.copy_(q_eual_param.data * tau + (1 - tau) * q_target_param)


    def choose_action(self, state:np.ndarray, isTrain=False):
        state = torch.tensor([state], dtype=torch.float).to(device)
        action_q = self.eval_nn.forward(state)
        action = torch.argmax(action_q, dim=-1).item()

        if np.random.rand() < self.epsilon and isTrain:
            action = np.random.choice(self.action_dim)
        
        return action
        

    def remember(self, state, action, reward, next_state, terminal):
        self.replay_buffer.store_once(state, action, reward, next_state, terminal)

    def learn(self):
        if not self.replay_buffer.ready():
            return

        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self.replay_buffer.sample_buffer()
        batch_idx = np.arange(self.batch_size)

        state_batch = torch.tensor(state_batch, dtype=torch.float).to(device)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float).to(device)
        next_state_batch = torch.tensor(next_state_batch, dtype=torch.float).to(device)
        terminal_batch = torch.tensor(terminal_batch, dtype=torch.bool).to(device)

        with torch.no_grad():
            q_ns = torch.max(self.target_nn.forward(next_state_batch), dim=-1)[0]
            q_ns[terminal_batch] = 0
            y_ = reward_batch + self.gamma * q_ns
            
        y = self.eval_nn.forward(state_batch)[batch_idx, action_batch]

        loss = F.mse_loss(y, y_.detach())
        self.eval_nn.optimizer.zero_grad()
        loss.backward()
        self.eval_nn.optimizer.step()

        self.update_network_parameters()

        if self.epsilon > self.eps_min:
            self.epsilon -= self.eps_dec
        else:
            self.epsilon = self.eps_min
