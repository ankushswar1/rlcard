''' An example of learning a Deep-Q Agent on UNO
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)  # or any {DEBUG, INFO, WARN, ERROR, FATAL}
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # or any {DEBUG, INFO, WARN, ERROR, FATAL}



import rlcard
from rlcard.agents import DQNAgent
from rlcard.agents import RandomAgent
from rlcard.utils.utils import set_global_seed, tournament
from rlcard.utils import Logger
import csv

# Make environment
env = rlcard.make('uno', config={'seed': 0})
eval_env = rlcard.make('uno', config={'seed': 0})

# Set the iterations numbers and how frequently we evaluate the performance
evaluate_every = 1
evaluate_num = 500
episode_num = 100

# The intial memory size
memory_init_size = 1000

# Train the agent every X steps
train_every = 1

# The paths for saving the logs and learning curves
log_dir = './experiments/uno_dqn_result/'

# Set a global seed
set_global_seed(0)

data = []

with tf.Session() as sess:

    with open(log_dir + 'A1.csv', 'w') as f:
        csvw = csv.writer(f)
        # Initialize a global step
        global_step = tf.Variable(0, name='global_step', trainable=False)

        # Save model
        save_dir = 'models/uno_dqn'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        saver = tf.train.Saver()

        # Set up the agents
        agent = DQNAgent(sess,
                        scope='dqn',
                        action_num=env.action_num,
                        replay_memory_size=20000,
                        replay_memory_init_size=memory_init_size,
                        train_every=train_every,
                        state_shape=env.state_shape,
                        mlp_layers=[512,512])
        random_agent1 = RandomAgent(action_num=eval_env.action_num)
        random_agent2 = RandomAgent(action_num=eval_env.action_num)
        env.set_agents([agent, random_agent1, random_agent2])
        eval_env.set_agents([agent, random_agent1, random_agent2])

        # Initialize global variables
        sess.run(tf.global_variables_initializer())

        # Init a Logger to plot the learning curve
        logger = Logger(log_dir)

        for episode in range(episode_num):
            print('Episode: ' + str(episode))

            # Generate data from the environment
            trajectories, _ = env.run(is_training=True)

            # Feed transitions into agent memory, and train the agent
            for ts in trajectories[0]:
                agent.feed(ts)

            # Evaluate the performance. Play with random agents.
            if episode % evaluate_every == 0:
                a, b = env.timestep, tournament(eval_env, evaluate_num)[0]
                logger.log_performance(a, b)
                csvw.writerow([a,b])
                f.flush()
                
        # Close files in the logger
        logger.close_files()

        # Plot the learning curve
        logger.plot('DQN')
        saver.save(sess, os.path.join(save_dir, 'model_FINAL'))

    

    
    
