from ReplayBuffer import *
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
'''
FCN要素
--------
-全链接网络结构
--各层维度，学习率，优化器（__init__）
-网络传播
--前向传播，激活函数（forward）
'''
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


class DDQN():
    def __init__(self, action_dim, state_dim, hidden_dim1, hidden_dim2, 
                 alpha, tau, gamma, epsilon, eps_min, eps_des,
                 max_size, batch_size) -> None:
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.alpha = alpha
        self.tau = tau
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_des = eps_des
        self.max_size = max_size
        self.batch_size = batch_size

        self.eval_nn = FCN(self.alpha, self.state_dim, self.action_dim, self.hidden_dim1, self.hidden_dim2)
        self.target_nn = FCN(self.alpha, self.state_dim, self.action_dim, self.hidden_dim1, self.hidden_dim2)

        self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim, self.max_size, self.batch_size)
        