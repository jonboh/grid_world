from itertools import product

import keras as k
from td_learning import *


def net_model(n_dense, n_representation):
    dense_L1 = k.layers.Dense(units=n_dense)
    dense_out = k.layers.Dense(units=1)

    state_action_representation = k.layers.Input(shape=(n_representation,))
    x = dense_L1(state_action_representation)
    value = dense_out(x)

    model = k.models.Model(inputs=state_action_representation, outputs=value)
    return model


if __name__ == '__main__':
    episode_length = 15
    learning_rate = 1 / episode_length ** 2
    environment = env.ClassicNonDeterministicGridWorld(food=1, death=-1, discount=0.9, penalty=-0.1)
    # environment = env.ClassicGridWorld(food=1, death=-1, discount=0.9, penalty=-0.1)

    state_action_set = tuple(product(environment.states, environment.agent.actions))
    state_action2vec = dict()
    count = 0
    for state_action in state_action_set:
        state_action2vec[state_action] = k.utils.to_categorical(count, len(state_action_set))
        count += 1
    value_net = net_model(10, len(state_action_set))
    value_net.summary()
    lr = 0.001
    optim = k.optimizers.Adam(lr=lr)
    value_net.compile(optimizer=optim, loss='mean_squared_error', metrics=['accuracy'])

    state_action_rep = np.zeros((4, 55))
    value_net.predict(x=[state_action_rep])

    td_agent = TDNeuralAgent(0.7, learning_rate, environment, value_net, state_action2vec)
    n_batch = 100
    n_episode_batch = 10
    count = 0
    for i in range(n_batch):
        for j in range(n_episode_batch):
            episode = td_agent.play_exploratory_episode(episode_length, 0.03)
            count += 1
        td_agent.process_episodes()
        episode = td_agent.play_greedy_episode(episode_length)
        td_agent.print_policy()
        print('Greedy Episode: {0}  Reward: {1:4.2f}'.format(i, sum(episode.rewards)))
        episode.print_action_sequence()
        # input()
    episode = td_agent.play_greedy_episode(10)
    print(episode)
