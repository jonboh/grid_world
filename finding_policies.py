from environment import *

# Value Iteration
environment = ClassicGridWorld(food=1, death=-10, penalty=-0.1, discount=0.9)
# Initialize U
u_dict = dict()
for state in environment.states:
    u_dict.update({state: 0})

actions = environment.agent.actions
discount = environment.agent.discount
penalty = environment.agent.penalty

n_iterations = 1000
for _ in range(n_iterations):
    for state in u_dict:
        max_utility = -10000
        max_reward = None
        for action in actions:
            new_state, new_reward = state.do_action(action)
            utility = u_dict[new_state] + new_reward + penalty
            if utility > max_utility:
                max_utility = utility
                max_reward = new_reward
        u_dict[state] = discount * max_utility

for state in u_dict:
    print('{0} : {1:8.2f}'.format(str(state), u_dict[state]))