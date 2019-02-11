import numpy as np


class resdict(dict):
    def __init__(self, default_out=None, *args, **kargs):
        super().__init__(*args, **kargs)
        self.default_out = default_out

    def __missing__(self, key):
        return self.default_out

    def update(self, dictionary):  # update will only work with dictionaries
        super().update(dictionary)
        if type(dictionary) is resdict:
            self.default_out = dictionary.default_out


class State:
    def __init__(self, name, action2state=None, action2reward=None):
        self.prob_tolerance = 0.05
        self.name = name
        self.action2state = resdict(((self,), (1,)))  # default outcome is itself (void or invalid action)
        if action2state is not None:
            self.define_action2state(action2state)
        self.action2reward = resdict(((0,), (1,)))  # default reward is 0
        if action2reward is not None:
            self.define_action2reward(action2reward)

    def define_action2state(self, action2state: dict):
        # Check Input
        try:
            for key, entry in action2state.items():
                if len(entry) != 2:
                    raise TypeError('Wrong action2state definition in {0}'.format(self.name))
                elif len(entry[0]) != len(entry[1]):
                    raise TypeError('Wrong action2state definition in {0}'.format(self.name))
                elif type(entry[0]) != tuple or type(entry[1]) != tuple:
                    raise TypeError('Wrong action2state definition in {0}'.format(self.name))
                elif 1.0 - self.prob_tolerance > sum(entry[1]) or sum(entry[1]) > 1.0 + self.prob_tolerance:
                    raise TypeError('Wrong action2state definition in {0}. '
                                    'Probabilities must sum 1'.format(self.name))
        except:
            raise TypeError('Wrong action2state definition in {0}'.format(self.name))
        self.action2state.update(action2state)

    def define_action2reward(self, action2reward: dict):
        # Check Input
        try:
            for key, entry in action2reward.items():
                if len(entry) != 2:
                    raise TypeError('Wrong action2reward definition in {0}'.format(self.name))
                elif len(entry[0]) != len(entry[1]):
                    raise TypeError('Wrong action2reward definition in {0}'.format(self.name))
                elif type(entry[0]) != tuple or type(entry[1]) != tuple:
                    raise TypeError('Wrong action2reward definition in {0}'.format(self.name))
                elif 1.0 - self.prob_tolerance > sum(entry[1]) > 1.0 + self.prob_tolerance:
                    raise TypeError('Wrong action2reward definition in {0}. '
                                    'Probabilities must sum 1'.format(self.name))
        except:
            raise TypeError('Wrong action2state definition in {0}'.format(self.name))
        self.action2reward.update(action2reward)

    def random_selector(self, trans_entry):
        rand = np.random.random_sample(1)[0]
        cumprob = 0
        i = 0
        for prob in trans_entry[1]:
            cumprob += prob
            if rand < cumprob:
                return trans_entry[0][i]
            i += 1

    def trans_action2state(self, action):
        trans_entry = self.action2state[action]  # transition entry
        if len(trans_entry[0]) == 1:
            return trans_entry[0][0]
        else:  # random according to State Transition Model
            action = self.random_selector(trans_entry)
            return action

    def trans_action2reward(self, action):
        trans_entry = self.action2reward[action]  # transition entry
        if len(trans_entry[0]) == 1:
            return trans_entry[0][0]
        else:  # random according to Reward Transition Model
            reward = self.random_selector(trans_entry)
            return reward

    def do_action(self, action):
        new_state = self.trans_action2state(action)
        new_reward = self.trans_action2reward(action)
        return new_state, new_reward

    def __str__(self):
        return self.name


class Action:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name


class Agent:
    def __init__(self, name, state=None, discount=0.9, penalty=0.):
        self.name = name
        self.state = state
        self.reward = 0
        self.actions = list()
        self.discount = discount
        self.penalty = penalty

    def define_actions(self, actions: list):
        self.actions += actions

    def do_action(self, action):
        if action in self.actions:
            state, reward = self.state.do_action(action)
            self.state = state
            self.reward += self.discount * reward + self.penalty
        else:
            self.reward += self.penalty

    def __str__(self):
        string = 'Agent {0} now in state {1}, accumulated reward is {2:.2f}'.format(self.name, self.state, self.reward)
        return string


class ClassicGridWorld:
    def __init__(self, food=1, death=-1, discount=0.9, penalty=-0.01):
        up = Action(name='up')
        down = Action(name='down')
        left = Action(name='left')
        right = Action(name='right')
        void_action = Action(name='void')
        actions = [up, down, left, right, void_action]

        state00 = State(name='s00')
        state01 = State(name='s01')
        state02 = State(name='s02')
        state03 = State(name='s03')
        state10 = State(name='s10')
        state12 = State(name='s12')
        state13 = State(name='s13')
        state20 = State(name='s20')
        state21 = State(name='s21')
        state22 = State(name='s22')
        state23 = State(name='s23')
        self.states = [state00, state01, state02, state03, state10, state12, state13, state20, state21, state22,
                       state23]

        state00.define_action2state({up: ((state10,), (1,)), right: ((state01,), (1,))})
        state01.define_action2state({left: ((state00,), (1,)), right: ((state02,), (1,))})
        state02.define_action2state({up: ((state12,), (1,)), left: ((state01,), (1,)), right: ((state03,), (1,))})
        state03.define_action2state(resdict(((state00,), (1,))))
        state03.define_action2reward(resdict(((death,), (1,))))
        state10.define_action2state({up: ((state20,), (1,)), down: ((state00,), (1,))})
        state12.define_action2state({up: ((state22,), (1,)), down: ((state02,), (1,)), right: ((state13,), (1,))})
        state13.define_action2state({down: ((state03,), (1,)), left: ((state12,), (1,)), up: ((state23,), (1,))})
        state20.define_action2state({down: ((state10,), (1,)), right: ((state21,), (1,))})
        state21.define_action2state({left: ((state20,), (1,)), right: ((state22,), (1,))})
        state22.define_action2state({down: ((state12,), (1,)), left: ((state21,), (1,)), right: ((state23,), (1,))})
        state23.define_action2state(resdict(((state00,), (1,))))
        state23.define_action2reward(resdict(((food,), (1,))))

        self.initial_state = state00
        self.agent = Agent(name='billy', state=self.initial_state, discount=discount, penalty=penalty)
        self.agent.define_actions(actions)

    def reset(self):
        self.agent.state = self.initial_state
        self.agent.reward = 0


        self.initial_state = state00
        self.agent = Agent(name='billy', state=self.initial_state, discount=discount, penalty=penalty)
        self.agent.define_actions(actions)

    def reset(self):
        self.agent.state = self.initial_state
        self.agent.reward = 0


def select_action_helper(string, environment):
    if string == 'up':
        action = environment.agent.actions[0]
    elif string == 'down':
        action = environment.agent.actions[1]
    elif string == 'left':
        action = environment.agent.actions[2]
    elif string == 'right':
        action = environment.agent.actions[3]
    else:
        action = environment.agent.actions[4]
    return action


if __name__ == '__main__':

    environment = ClassicGridWorld()
    # Play
    print('Play! (Write the name of the action you want to take)')
    print('Type "exit" to quit')
    input_str = ''
    while input_str != 'exit':
        print(environment.agent)
        input_str = input()
        action = select_action_helper(input_str, environment)
        environment.agent.do_action(action)
