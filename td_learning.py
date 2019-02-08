import copy
import math
from itertools import product

import environment as env


class TDEpisode:
    def __init__(self, lenght):
        self.length = lenght
        self.states = [None for i in range(lenght)]
        self.rewards = [None for i in range(lenght)]

    def add_next_state_action_reward(self, state, action, reward):
        try:
            self.states[self.states.index(None)] = (state, action)
            self.rewards[self.rewards.index(None)] = reward
        except ValueError:
            raise EOFError('Episode is already filled')

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

    def search_max_reward_action(self, state):
        state_action_list = [(state, action) for action in self.environment.agent.actions]
        max_state_action = None
        max_value = -math.inf
        for state_action in state_action_list:
            if self.value_table[state_action] > max_value:
                max_state_action = state_action
                max_value = self.value_table[state_action]
        return max_state_action[1]

    def play_greedy_episode(self, episode_lenght, return_seq=False):
        episode = TDEpisode(episode_lenght)
        for t in range(episode_lenght):
            state = environment.agent.state
            action = self.search_max_reward_action(state)
            environment.agent.do_action(action)
            episode.add_next_state_action_reward(state, action, environment.agent.reward)
            environment.agent.reward = 0  # otherwise it will be accumulated
        return episode

    def process_episode(self, episode):
        prev_value_table = copy.copy(self.value_table)
        eligibility = dict.fromkeys(self.value_table.keys(), 0)
        for t in range(episode.length):
            eligibility[episode.states[t]] = eligibility[episode.states[t]] + 1
            for state_action in self.value_table.keys():
                if eligibility[state_action] == 0:
                    continue
                else:
                    self.value_table[state_action] = self.value_table[state_action] + self.learn_rate * (
                            episode.rewards[t] +
                            self.environment.agent.discount * prev_value_table[episode.states[t]] -
                            prev_value_table[episode.states[t]]
                    ) * eligibility[state_action]
                    eligibility[state_action] = self.td_lambda * self.environment.agent.discount * \
                                                eligibility[state_action]

    def process_episode_tdlambda0(self, episode):
        prev_value_table = copy.copy(self.value_table)
        for t in range(episode.length):
            for state_action in self.value_table.keys():
                self.value_table[state_action] = self.value_table[state_action] + self.learn_rate * (
                        episode.rewards[t] +
                        self.environment.agent.discount * prev_value_table[episode.states[t]] -
                        prev_value_table[episode.states[t]])



    def process_episode_tdlambda1(self, episode):
        prev_value_table = copy.copy(self.value_table)
        eligibility = dict.fromkeys(self.value_table.keys(), 0)
        for t in range(episode.length):
            eligibility[episode.states[t]] = eligibility[episode.states[t]] + 1
            for state_action in self.value_table.keys():
                if eligibility[state_action] == 0:
                    continue
                elif t!=0:
                    self.value_table[state_action] = self.value_table[state_action] + self.learn_rate * (
                            episode.rewards[t] +
                            self.environment.agent.discount * prev_value_table[episode.states[t]] -
                            prev_value_table[episode.states[t-1]]
                    ) * eligibility[state_action]
                    eligibility[state_action] = self.environment.agent.discount * eligibility[state_action]
                else:
                    self.value_table[state_action] = self.value_table[state_action] + self.learn_rate * (
                            episode.rewards[t] +
                            self.environment.agent.discount * prev_value_table[episode.states[t]]
                    ) * eligibility[state_action]
                    eligibility[state_action] = self.environment.agent.discount * eligibility[state_action]



def value_table_print(value_table):
    for key, value in value_table.items():
        print('{0}, {1:5}: {2:7.4f}'.format(key[0], key[1].__str__(), value))

if __name__ == '__main__':
    environment = env.ClassicGridWorld()
    td_agent = TDAgent(0.7, 0.05, environment)
    n_episodes = 100
    for i in range(n_episodes):
        print('Episode: {0}'.format(i))
        episode = td_agent.play_greedy_episode(10)
        td_agent.process_episode(episode)
        environment.reset()
        # value_table_print(td_agent.value_table)
        # input()
    episode = td_agent.play_greedy_episode(10)
    print(episode)
