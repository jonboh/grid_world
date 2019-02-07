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
        self.name = name
        self.action2state = resdict(self)
        if action2state is not None:
            self.define_action2state(action2state)
        self.action2reward = resdict(0)
        if action2reward is not None:
            self.define_action2reward(action2reward)

    def define_action2state(self, action2state: dict):
        self.action2state.update(action2state)

    def define_action2reward(self, action2reward: dict):
        self.action2reward.update(action2reward)

    def do_action(self, action):
        new_state = self.action2state[action]
        new_reward = self.action2reward[action]
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

        state00.define_action2state({up: state10, right: state01})
        state01.define_action2state({left: state00, right: state02})
        state02.define_action2state({up: state12, left: state01, right: state03})
        state03.define_action2state(resdict(state00))
        state03.define_action2reward(resdict(death))
        state10.define_action2state({up: state20, down: state00})
        state12.define_action2state({up: state22, down: state02, right: state13})
        state13.define_action2state({down: state03, left: state12, up: state23})
        state20.define_action2state({down: state10, right: state21})
        state21.define_action2state({left: state20, right: state22})
        state22.define_action2state({down: state12, left: state21, right: state23})
        state23.define_action2state(resdict(state00))
        state23.define_action2reward(resdict(food))

        self.agent = Agent(name='billy', state=state00, discount=discount, penalty=penalty)
        self.agent.define_actions(actions)


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
