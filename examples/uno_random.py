''' A toy example of playing Uno with random agents
'''

import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import set_global_seed
import csv

# Make environment
env = rlcard.make('uno', config={'seed': 0})
episode_num = 50

# Set a global seed
set_global_seed(0)


# state['obs'] = 7 of 4x15 arrays
# 4x15 array: 4 rows for 4 colors, 15 is card number
# 0-2: 0, 1, 2 cards for your player
# 3: target (top of deck)
# 4-6: 0, 1, 2 cards for other players
# state['legal_actions'] = array of legal actions

# action = just some number (encoding)

# Set up agents
agent_0 = RandomAgent(action_num=env.action_num)
agent_1 = RandomAgent(action_num=env.action_num)
agent_2 = RandomAgent(action_num=env.action_num)
# agent_3 = RandomAgent(action_num=env.action_num)
env.set_agents([agent_0, agent_1, agent_2])

with open('random_uno.csv', 'w', newline='') as f:
    w = csv.writer(f)
    for episode in range(episode_num):

        # Generate data from the environment
        trajectories, _ = env.run(is_training=False)

        # Print out the trajectories
        print('\nEpisode {}'.format(episode))
        for ts in trajectories[0]:
            w.writerow(ts)
            # print(ts[4])
            # print('State: {}, Action: {}, Reward: {}, Next State: {}, Done: {}'.format(ts[0], ts[1], ts[2], ts[3], ts[4]))
