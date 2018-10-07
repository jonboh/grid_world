import numpy as np
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
                max_utility = -10000
                for action in actions:
                    new_state, new_reward = state.do_action(action)
                    utility = new_reward + penalty + discount * self.utilty[new_state]
                    if utility > max_utility:
                        max_utility = utility
                self.utilty[state] = max_utility
        return self.utilty

if __name__ == '__main__':
    environment = ClassicGridWorld(food=1, death=-10, penalty=-0.1, discount=0.9)
    solver = ValueIteration(environment)
    utility = solver.iterate(n_iterations=1000)
    for state in utility:
        print('{0} : {1:8.2f}'.format(str(state), utility[state]))