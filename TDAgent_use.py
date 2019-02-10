from td_learning import *


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
