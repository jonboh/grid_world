
class resdict(dict):
    def __init__(self, default_out=None, *args, **kargs):
        super().__init__(*args, **kargs)
        self.default_out = default_out

    def __missing__(self, key):
        return self.default_out

    def update(self, dictionary):
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


class Agent:
    def __init__(self, name, state=None, discount=0.9, penalty=0.):
        self.name = name
        self.state = state
        self.reward = 0
        self.actions = [void_action]
        self.discount = discount
        self.penalty = penalty

    def define_actions(self, actions: list):
        self.actions.append(actions)

    def do_action(self, action):
        state, reward = self.state.do_action(action)
        self.state = state
        self.reward += self.discount * reward - self.penalty

    def __str__(self):
        string = 'Agent {0} now in state {1}, accumulated reward is {2}'.format(self.name, self.state, self.reward)
        return string


def select_action(string):
    if string == 'up':
        action = up
    elif string == 'down':
        action = down
    elif string == 'left':
        action = left
    elif string == 'right':
        action = right
    else:
        action = void_action
    return action


if __name__ == '__main__':
    # Define Environment
    void_action = Action(name='void')
    up = Action(name='up')
    down = Action(name='down')
    left = Action(name='left')
    right = Action(name='right')

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
    state30 = State(name='s30')
    state31 = State(name='s31')
    state32 = State(name='s32')
    state33 = State(name='s33')

    food = +1
    death = -1
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

    agent = Agent(name='billy', state=state22, discount=0.9, penalty=0.01)
    agent.define_actions([up, down, left, right])

    # Play
    print('Play! (Write the name of the action you want to take)')
    input_str = ''
    while input_str != 'exit':
        print(agent)
        input_str = input()
        action = select_action(input_str)
        agent.do_action(action)
