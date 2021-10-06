from replay_memory import ReplayBuffer
from dueling_deep_q_network import DuelingDeepQNetwork
import numpy as np
import torch as T


class DuelingDQNAgent:
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
        self.Q_eval = DuelingDeepQNetwork(self.lr, n_actions=n_actions,
                                   input_dims=self.input_dims,
                                   fc1_dims=256, fc2_dims=256,
                                   name=self.env_name+'_'+self.algo+'_q_eval',
                                   chkpt_dir=self.chkpt_dir)
        self.Q_next = DuelingDeepQNetwork(self.lr, n_actions=n_actions,
                                   input_dims=self.input_dims,
                                   fc1_dims=256, fc2_dims=256,
                                   name=self.env_name + '_' + self.algo + '_q_next',
                                   chkpt_dir=self.chkpt_dir)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation]).to(self.Q_eval.device)
            _, advantage = self.Q_eval.forward(state)
            action = T.argmax(advantage).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

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

        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()

        indices = np.arange(self.batch_size)

        V_s, A_s = self.Q_eval.forward(states)
        V_s_, A_s_ = self.Q_eval.forward(states_)

        q_pred = T.add(V_s,
                       (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]
        q_next = T.add(V_s_,
                       (A_s_ - A_s_.mean(dim=1, keepdim=True))).max(dim=1)[0]
        q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next

        loss = self.Q_eval.loss(q_target, q_pred).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()
        self.learn_step_counter += 1
        self.decrement_epsilon()
