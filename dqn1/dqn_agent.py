from replay_memory import ReplayBuffer
from deep_q_network import DeepQNetwork
import numpy as np
import torch as T


class DQNAgent:
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
                 max_mem_size=100000, eps_end=0.01, eps_dec=5e-4, replace=1000,
                 algo=None, env_name=None, chkpt_dir='tmp/dqn'):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.input_dims = input_dims
        self.lr = lr
        self.n_actions = n_actions
        self.action_space = [i for i in range(self.n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        # counter!
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.learn_step_counter = 0
        self.memory = ReplayBuffer(max_mem_size, input_dims, n_actions)
        self.Q_eval = DeepQNetwork(self.lr, n_actions=n_actions,
                                   input_dims=self.input_dims,
                                   fc1_dims=256, fc2_dims=256,
                                   name=self.env_name+'_'+self.algo+'_q_eval',
                                   chkpt_dir=self.chkpt_dir)
        self.Q_next = DeepQNetwork(self.lr, n_actions=n_actions,
                                   input_dims=self.input_dims,
                                   fc1_dims=256, fc2_dims=256,
                                   name=self.env_name + '_' + self.algo + '_q_next',
                                   chkpt_dir=self.chkpt_dir)

        # self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        # self.new_state_memory = np.zeros((self.mem_size, *input_dims),
        #                                  dtype=np.float32)
        # self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        # self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        # self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)
    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)
        # index = self.mem_cntr % self.mem_size
        # self.state_memory[index] = state
        # self.new_state_memory[index] = state_
        # self.reward_memory[index] = reward
        # self.action_memory[index] = action
        # self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_memory(self):
        state, action, reward, new_state, done = \
                                                self.memory.sample_buffer(self.batch_size)

        states = T.tensor(state).to(self.Q_eval.device)
        rewards = T.tensor(reward).to(self.Q_eval.device)
        dones = T.tensor(done).to(self.Q_eval.device)
        actions = T.tensor(action).to(self.Q_eval.device)
        states_ = T.tensor(new_state).to(self.Q_eval.device)

        return states, actions, rewards, states_, dones

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.Q_next.load_state_dict(self.Q_eval.state_dict())
        return

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
                        if self.epsilon > self.eps_min else self.eps_min

        return

    def save_models(self):
        self.Q_eval.save_checkpoint()
        self.Q_next.save_checkpoint()

    def load_models(self):
        self.Q_eval.load_checkpoint()
        self.Q_next.load_checkpoint()

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return
        self.Q_eval.optimizer.zero_grad()

        # max_mem = min(self.mem_cntr, self.mem_size)

        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()

        indices = np.arange(self.batch_size)
        q_pred = self.Q_eval.forward(states)[indices, actions]
        q_next = self.Q_next.forward(states_).max(dim=1)[0]
        q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next
        # batch = np.random.choice(max_mem, self.batch_size, replace=False)
        #
        # batch_index = np.arange(self.batch_size, dtype=np.int32)

        # state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        # new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        # reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        # terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
        #
        # action_batch = self.action_memory[batch]
        #
        # # now we perform feedforward
        # q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        # q_next = self.Q_eval.forward(new_state_batch)
        # # q_next is target network
        # q_next[terminal_batch] = 0.0

        # q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]
        # T.max is indexed 0 because returns a tuple (value, index)

        loss = self.Q_eval.loss(q_target, q_pred).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()
        self.learn_step_counter += 1
        self.decrement_epsilon()
