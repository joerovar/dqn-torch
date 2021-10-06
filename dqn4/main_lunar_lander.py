import gym
from dueling_ddqn_agent import DuelingDDQNAgent
from utils import plot_learning
import numpy as np

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    best_score = -np.inf
    agent = DuelingDDQNAgent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4,
                            eps_min=0.1, input_dims=[8], lr=0.001, mem_size=50000,
                            chkpt_dir='models/', algo='DuelingDDQNAgent', env_name='LunarLander-v2')
    scores, eps_history, steps_array = [], [], []
    load_checkpoint = False
    n_games = 400

    if load_checkpoint:
        agent.load_models()

    fname = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) + '_' + \
        '_' + str(n_games) + 'games'
    figure_file = 'plots/' + fname + '.png'
    n_steps = 0

    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward

            if not load_checkpoint:
                agent.store_transition(observation, action, reward,
                                       observation_, int(done))
                agent.learn()
            observation = observation_
            n_steps += 1
        scores.append(score)
        steps_array.append(n_steps)

        avg_score = np.mean(scores[-100:])
        print('episode ', i, 'score %.2f' % score, 'average score %.2f' % avg_score,
              'epsilon %.2f' % agent.epsilon, 'steps ', n_steps)
        if avg_score > best_score:
            if not load_checkpoint:
                agent.save_models()
            best_score = avg_score

        eps_history.append(agent.epsilon)

    plot_learning(steps_array, scores, eps_history, figure_file)