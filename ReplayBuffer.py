import torch
import numpy as np

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size, batch_size) -> None:
        '''
        传入状态维度state_dim，动作维度action_dim，replaybuffer最大值max_size，每次取出的数据数量batch_size
        '''
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_size = max_size
        self.batch_size = batch_size

        # input
        self.state_buffer = np.zeros((self.max_size, self.state_dim))
        self.action_buffer = np.zeros((self.max_size, ))
        # output
        self.reward_buffer = np.zeros((self.max_size, ))
        self.next_state_buffer = np.zeros((self.max_size, self.state_dim))
        self.terminal_buffer = np.zeros((self.max_size,), dtype=np.bool)

        self.buffer_cnt = 0

    def store_once(self, state, action, reward, next_state, terminal):
        self.buffer_cnt = self.buffer_cnt % self.max_size

        self.state_buffer[self.buffer_cnt] = state
        self.action_buffer[self.buffer_cnt] = action
        self.reward_buffer[self.buffer_cnt] = reward
        self.next_state_buffer[self.buffer_cnt] = next_state
        self.terminal_buffer[self.buffer_cnt] = terminal

        self.buffer_cnt += 1

    def sample_buffer(self):
        '''
        从replay_buffer中随机采样batch_size个数据用来训练
        
        Args:
            None
        
        Returns:
            Tuple[state_batch, action_batch, reward_batch, next_state_batch, terminal_batch]

        '''
        buffer_len = min(self.buffer_cnt, self.max_size)

        idx_batch = np.random.choice(buffer_len, self.batch_size, replace=False)

        state_batch = self.state_buffer[idx_batch]
        action_batch = self.action_buffer[idx_batch]
        reward_batch = self.reward_buffer[idx_batch]
        next_state_batch = self.next_state_buffer[idx_batch]
        terminal_batch = self.terminal_buffer[idx_batch]

        return state_batch, action_batch, reward_batch, next_state_batch, terminal_batch
    
    def ready(self):
        return self.buffer_cnt > self.batch_size