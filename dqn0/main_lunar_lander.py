import gym
from dqn_agent import DQNAgent
from utils import plot_learning
import numpy as np

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    agent = DQNAgent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4,
                     eps_min=0.1, input_dims=[8], lr=0.001)
    scores, eps_history, steps_array = [], [], []
    n_games = 400
    n_steps = 0
    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward,
                                   observation_, done)
            agent.learn()
            observation = observation_
            # n_steps += 1
        scores.append(score)
        eps_history.append(agent.epsilon)
        # steps_array.append(n_steps)
        avg_score = np.mean(scores[-100:])
        print('episode ', i, 'score %.2f' % score, 'average score %.2f' % avg_score,
              'epsilon %.2f' % agent.epsilon)

    x = [i+1 for i in range(n_games)]
    filename = 'lunar_lander_2020_dqn0.png'
    plot_learning(x, scores, eps_history, filename)

