from ReplayBuffer import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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

'''
DDQN设计要素
------------
-网络更新
--将eval_nn网络的参数更新给target_nn（update_network_parameters）
-从replay_buffer中拿出经验数据进行学习
--拿出数据学习（learn）
'''
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


    def update_network_parameters(self, tau = None):
        if tau == None:
            tau = self.tau
        
        for eval_nn_param, target_nn_param in zip(self.eval_nn.parameters(), self.target_nn.parameters()):
            target_nn_param.copy_(eval_nn_param * tau + target_nn_param * (1 - tau))  

    def learn(self):
        if not self.replay_buffer.ready():
            return
        
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self.replay_buffer.sample_buffer()
        batch_idx = np.arange(self.batch_size)

        state_batch = torch.tensor(state_batch, torch.float).to(device)
        # action_batch = torch.tensor(action_batch, torch.int).to(device)
        reward_batch = torch.tensor(reward_batch, torch.float).to(device)
        next_state_batch = torch.tensor(next_state_batch, torch.float).to(device)
        terminal_batch = torch.tensor(terminal_batch, torch.bool).to(device)
# ------------------------------------------------------------------------------------------
# 尝试写DDQN的主要逻辑
        y = self.eval_nn.forward(state_batch)[batch_idx, action_batch]

        with torch.no_grad():
            action_eval = torch.max(self.eval_nn.forward(next_state_batch), dim=-1)[1]
            qn_ae = self.target_nn.forward(next_state_batch)[batch_idx, action_eval]
            qn_ae[terminal_batch] = 0
            y_ = reward_batch + self.gamma * qn_ae

# -------------------------------------------------------------------------------------------
# 不熟悉
        loss = F.mse_loss(y, y_.detach())
        self.eval_nn.optimizer.zero_grad()
        loss.backward()
        self.eval_nn.optimizer.step()

