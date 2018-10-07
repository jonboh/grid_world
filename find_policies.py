import numpy as np
import math
from environment import *


class ValueIteration:
    def __init__(self, environment, utility=None):
        self.environment = environment
        if utility is None:
            self.utilty = dict()
            for state in environment.states:
                rand = np.random.random()
                self.utilty.update({state: rand})
        else:
            # check_environment_utility_are_compatible() UNIMPLEMENTED
            self.utilty = utility

    def iterate(self, n_iterations):
        actions = environment.agent.actions
        discount = environment.agent.discount
        penalty = environment.agent.penalty
        for _ in range(n_iterations):
            for state in self.utilty:
                max_utility = -math.inf
                for action in actions:
                    new_state, new_reward = state.do_action(action)
                    utility = new_reward + penalty + discount * self.utilty[new_state]
                    if utility > max_utility:
                        max_utility = utility
                self.utilty[state] = max_utility
        return self.utilty


class PolicyIteration:
    def __init__(self, environment, utility=None, policy=None):
        self.environment = environment
        actions = self.environment.agent.actions
        states = self.environment.states
        if utility is None:
            self.utilty = dict()
            for state in environment.states:
                rand = np.random.random()
                self.utilty.update({state: rand})
        else:
            # check_environment_utility_are_compatible() UNIMPLEMENTED
            self.utilty = utility
        if policy is None:
            self.policy = dict()
            randint = np.random.randint(0, len(actions), size=len(states))
            for state, i in zip(states, randint):
                self.policy.update({state: actions[i]})
        else:
            self.policy = policy

    def iterate(self, n_iterations):
        states = self.environment.states
        discount = self.environment.agent.discount
        actions = self.environment.agent.actions
        for _ in range(n_iterations):
            # Evaluate Policy
            for state in utility:
                new_state, new_reward = state.do_action(self.policy[state])
                utility[state] = new_reward + discount * utility[new_state]
            # Improve Policy
            for state in self.policy:
                max_action = -math.inf
                max_utility = -math.inf
                for action in actions: # argmax( T(s,a,s') * U(s')
                    new_state, new_reward = state.do_action(action)
                    if max_utility < utility[new_state]:
                        max_utility = utility[new_state]
                        max_action = action
                self.policy[state] = max_action
        return self.policy

if __name__ == '__main__':
    environment = ClassicGridWorld(food=1, death=-1, penalty=-0.1, discount=0.9)

    print('Value Iteration: ')
    solver = ValueIteration(environment)
    utility = solver.iterate(n_iterations=100)
    for state in utility:
        print('{0} : {1:8.2f}'.format(str(state), utility[state]))

    print('\n\n')
    print('Policy Iteration')
    solver = PolicyIteration(environment)
    policy = solver.iterate(n_iterations=100)
    for state in policy:
        print('{0} : {1:.2f} => {2}'.format(state, solver.utilty[state], policy[state]))
