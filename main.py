import random
from collections import deque
import numpy as np
import gym
import torch
from torch.autograd import Variable
from model import Learner
# import matplotlib.pyplot as plt


class DQNAgent(object):

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = Learner(state_size, action_size, self.learning_rate)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values.data.numpy()[0])

    def replay(self, batch_size):
        batch_size = batch_size if len(self.memory) >= batch_size\
                                else len(self.memory)
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target = reward + self.gamma *\
                            self.model.predict(next_state).max()
                target = target.data.numpy()
            target_f = self.model.predict(state).data.numpy()
            target_f[0][action] = target
            self.model.fit(Variable(torch.Tensor(state)),
                           Variable(torch.Tensor(target_f)))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def main():
    scores = []

    env = gym.make('CartPole-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)
    done = False
    batch_size = 100
    episodes = 300

    for e in range(episodes):

        state = env.reset()
        state = np.reshape(state, [1, state_size])

        for time_t in range(500):
            # env.render()

            action = agent.act(state)

            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])

            agent.remember(state, action, reward, next_state, done)

            state = next_state

            if done:
                print('episode: {}/{}, score: {}'.format(e, episodes, time_t))
                scores.append(time_t)
                break

        agent.replay(batch_size)

    # plt.title('score')
    # plt.xlabel('episode')
    # plt.ylabel('score')
    # plt.plot(scores)
    # plt.show()

    state = env.reset()
    state = np.reshape(state, [1, state_size])
    for time_t in range(500):
        env.render()
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, state_size])
        state = next_state
        if done:
            break


if __name__ == '__main__':
    main()
