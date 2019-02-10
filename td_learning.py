import copy
import math
from itertools import product
import numpy as np

import environment as env


class TDEpisode:
    def __init__(self, lenght):
        self.length = lenght
        self.states = [None for _ in range(lenght)]
        self.rewards = [None for _ in range(lenght)]

    def add_next_state_action_reward(self, state, action, reward):
        try:
            self.states[self.states.index(None)] = (state, action)
            self.rewards[self.rewards.index(None)] = reward
        except ValueError:
            raise EOFError('Episode is already filled')

    def print_action_sequence(self):
        str_prt = ', '.join(map(lambda x: str(x[1]), self.states))
        print(str_prt)

    def __str__(self):
        str_ret = ''
        for state_action in self.states:
            str_ret += 'State {0}, Action {1}\n'.format(state_action[0], state_action[1])
        return str_ret


class TDAgent:
    def __init__(self, td_lambda, learn_rate, environment):
        self.td_lambda = td_lambda
        self.learn_rate = learn_rate
        self.environment = environment
        # value_table( state space ) = cartesianproduct(states, actions)
        self.value_table = dict.fromkeys(product(environment.states, environment.agent.actions), 0)
        self.episodes = list()
        self.unprocessed_episodes = list()

    def search_max_reward_action(self, state):
        state_action_list = [(state, action) for action in self.environment.agent.actions]
        max_state_action = None
        max_value = -math.inf
        for state_action in state_action_list:
            if self.value_table[state_action] > max_value:
                max_state_action = state_action
                max_value = self.value_table[state_action]
        return max_state_action[1]

    def play_exploratory_episode(self, episode_lenght, explore_rate):
        episode = TDEpisode(episode_lenght)
        rand_vect = np.random.randn(episode_length)
        rand_index = np.random.randint(0, len(environment.agent.actions), episode_length)  # not all will be used
        for t in range(episode_lenght):
            state = environment.agent.state
            if rand_vect[t] >= explore_rate:
                action = self.search_max_reward_action(state)
            else:
                action = environment.agent.actions[rand_index[t]]
            environment.agent.do_action(action)
            episode.add_next_state_action_reward(state, action, environment.agent.reward)
            environment.agent.reward = 0  # otherwise it will be accumulated
        environment.reset()
        self.unprocessed_episodes.append(episode)
        return episode

    def play_greedy_episode(self, episode_lenght):
        episode = TDEpisode(episode_lenght)
        for t in range(episode_lenght):
            state = environment.agent.state
            action = self.search_max_reward_action(state)
            environment.agent.do_action(action)
            episode.add_next_state_action_reward(state, action, environment.agent.reward)
            environment.agent.reward = 0  # otherwise it will be accumulated
        environment.reset()
        return episode

    def max_value_next_state(self, state):
        max_action = None
        max_value = -math.inf
        for action in self.environment.agent.actions:
            if self.value_table[(state, action)] > max_value:
                max_action = action
                max_value = self.value_table[(state, action)]
        return max_value

    def process_episode(self, episode):
        prev_value_table = copy.copy(self.value_table)
        eligibility = dict.fromkeys(self.value_table.keys(), 0)
        for t in range(episode.length):
            eligibility[episode.states[t]] = eligibility[episode.states[t]] + 1
            for state_action in self.value_table.keys():
                if t == episode.length - 1:
                    break
                if eligibility[state_action] == 0:
                    continue
                else:
                    self.value_table[state_action] = self.value_table[state_action] + self.learn_rate * (
                            episode.rewards[t] +
                            self.environment.agent.discount * self.max_value_next_state(episode.states[t + 1][0]) -
                            prev_value_table[episode.states[t]]
                    ) * eligibility[state_action]
                    eligibility[state_action] = self.td_lambda * self.environment.agent.discount * \
                                                eligibility[state_action]

    def process_episodes(self):
        for episode in self.unprocessed_episodes:
            self.process_episode(episode)
        self.unprocessed_episodes = list()


def value_table_print(value_table):
    for key, value in value_table.items():
        print('{0}, {1:5}: {2:7.4f}'.format(key[0], key[1].__str__(), value))


if __name__ == '__main__':
    episode_length = 100
    learning_rate = 1 / episode_length ** 2
    environment = env.ClassicGridWorld(food=10, death=-5, discount=0.9, penalty=-0.1)
    td_agent = TDAgent(0.7, learning_rate, environment)
    n_batch = 1000
    n_episode_batch = 100
    count = 0
    for i in range(n_batch):
        for j in range(n_episode_batch):
            episode = td_agent.play_exploratory_episode(episode_length, 0.03)
            count += 1
        td_agent.process_episodes()
        episode = td_agent.play_greedy_episode(episode_length)
        value_table_print(td_agent.value_table)
        print('Greedy Episode: {0}  Reward: {1:4.2f}'.format(i, sum(episode.rewards)))
        episode.print_action_sequence()
        # input()
    episode = td_agent.play_greedy_episode(10)
    print(episode)
