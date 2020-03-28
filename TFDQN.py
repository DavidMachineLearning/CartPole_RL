from collections import deque
import tensorflow as tf
from time import sleep
import numpy as np
import random
import gym


def fully_connected(name, input_tensor, num_units, activation=tf.nn.relu):
    """Returns a fully connected layer"""
    # initialize weights
    w = tf.compat.v1.get_variable(f"W_{name}", shape=[input_tensor.get_shape()[1], num_units],
                                  initializer=tf.compat.v1.initializers.he_uniform(),
                                  dtype=tf.float32, trainable=True)
    # initialize bias
    b = tf.compat.v1.get_variable(f"B_{name}", shape=[num_units],
                                  initializer=tf.constant_initializer(0.0),
                                  dtype=tf.float32,
                                  trainable=True)
    # output
    out = tf.matmul(input_tensor, w) + b
    # add activation
    if activation:
        out = activation(out, name=f"activation_{name}")
    # change name
    out = tf.compat.v1.identity(out, name=name)

    return out


class DQNAgent:
    """Class providing DQN algorithm based on tensorflow"""
    def __init__(self, state_size, action_size, name, gamma=1.0, epsilon_start=1.0, epsilon_decay=0.0002,
                 epsilon_min=0.01, memorysize=100000, learning_rate=0.0001, n_Hlayers=2, n_nodes=64):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memorysize)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_max = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.step = 0
        self._build_model(name, n_Hlayers, n_nodes)
        
    def _build_model(self, name, hidden_layers, nodes):
        """Build the neural network structure"""
        with tf.variable_scope(name):
            self.inputs_ = tf.placeholder(tf.float32, [None, self.state_size], name='inputs')
            self.actions_ = tf.placeholder(tf.int32, [None], name='actions')
            one_hot_actions = tf.one_hot(self.actions_, self.action_size)
            self.targetQs_ = tf.placeholder(tf.float32, [None], name='target')
            self.layers = list()
            self.layers.append(fully_connected("hidden1", self.inputs_, nodes))
            for layer in range(hidden_layers):
                self.layers.append(fully_connected(f"hidden{layer+2}", self.layers[layer], nodes))
            self.output = fully_connected("output", self.layers[-1], self.action_size, activation=None)
            self.Q = tf.reduce_sum(tf.multiply(self.output, one_hot_actions), axis=1)
            self.loss = tf.reduce_mean(tf.square(self.targetQs_ - self.Q))
            self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def _take_action(self, state):
        """Return the action that gives the biggest reward"""
        feed = {self.inputs_: state.reshape((1, *state.shape))}
        Qs = sess.run(self.output, feed_dict=feed)
        return np.argmax(Qs)

    def _sample(self, batch_size):
        """Gets a random batch from the memory"""
        idx = np.random.choice(np.arange(len(self.memory)), 
                               size=batch_size, 
                               replace=False)
        return [self.memory[i] for i in idx]

    def remember(self, experience):
        """experience = (state, action, reward, next_state)"""
        self.memory.append(experience)
        
    def action(self, state, mode='train'):
        """Return an action based on the current state, if mode is test the agent choose
        the best action that gives the biggest reward.
        If mode is train, then it can choose also a random action with epsilon probability"""
        self.step += 1
        # reduce gradually epsilon to its minimum value
        self.epsilon = self.epsilon_min + (
            self.epsilon_max - self.epsilon_min)*np.exp(-self.epsilon_decay*self.step)
        if np.random.rand() > self.epsilon or mode.lower() == "test":
            return self._take_action(state)
        else:
            return random.randrange(self.action_size)
    
    def replay(self, batch_size):
        """Train the agent with a random batch of the collected data"""
        batch = self._sample(batch_size)
        states = np.array([each[0] for each in batch])
        actions = np.array([each[1] for each in batch])
        rewards = np.array([each[2] for each in batch])
        next_states = np.array([each[3] for each in batch])
        target_Qs = sess.run(self.output, feed_dict={self.inputs_: next_states})
        episode_ends = (next_states == np.zeros(states[0].shape)).all(axis=1)
        target_Qs[episode_ends] = (0, 0)
        
        targets = rewards + self.gamma * np.max(target_Qs, axis=1)

        loss, _ = sess.run([self.loss, self.opt], feed_dict={self.inputs_: states,
                                                             self.targetQs_: targets,
                                                             self.actions_: actions})
        return loss


def collect_experience(env_, agent_, size):
    """Save data in agent's memory, preparing it for training"""
    env_.reset()
    state, reward, done, _ = env_.step(env_.action_space.sample())
    for data in range(size):
        action = env_.action_space.sample()
        next_state, reward, done, _ = env_.step(action)
        # penalize reward based on the position of the cart
        reward = max(0, reward * (1 - abs(next_state[0]/2.4)))
        if done:
            next_state = np.zeros(state.shape)
            # save experience in agent's memory
            agent_.remember((state, action, reward, next_state))
            env_.reset()
            state, reward, done, _ = env_.step(env.action_space.sample())
        else:
            # save experience in agent's memory
            agent_.remember((state, action, reward, next_state))
            state = next_state

    
if __name__ == "__main__":
    # initialize gym environment and the agent
    env = gym.make('CartPole-v1')
    tf.reset_default_graph()
    agent = DQNAgent(4, 2, "main")
    train_episodes = 700
    batch_size = 32
    
    # populate memory
    collect_experience(env, agent, batch_size)
    
    # Iterate the game
    rewards_list = list()
    saver = tf.train.Saver()
    max_steps = 500
    with tf.Session() as sess:
        # Initialize variables
        sess.run(tf.compat.v1.global_variables_initializer())
        step = 0
        state = env.reset()
        for ep in range(1, train_episodes):
            total_reward = 0
            t = 0
            while t < max_steps:
                step += 1
                
                # Take action, get new state and reward
                action = agent.action(state)
                next_state, reward, done, _ = env.step(action)

                # penalize reward based on the position of the cart
                reward = max(0, reward * (1 - abs(next_state[0]/2.4)))

                # collect total reward
                total_reward += reward
                
                if done:
                    # the episode ends so no next state
                    next_state = np.zeros(state.shape)
                    t = max_steps
                    
                    print('Episode: {}'.format(ep),
                          'Total reward: {}'.format(total_reward),
                          'Explore P: {:.4f}'.format(agent.epsilon))

                    # Add reward to list
                    rewards_list.append(total_reward)
                    
                    # Add experience to memory
                    agent.remember((state, action, reward, next_state))
                    
                    # Start new episode
                    env.reset()
                    
                    # Take one random step to get the pole and cart moving
                    state, reward, done, _ = env.step(env.action_space.sample())

                else:
                    # Add experience to memory
                    agent.remember((state, action, reward, next_state))
                    state = next_state
                    t += 1

                # train using batch
                loss = agent.replay(batch_size)
                
            # if the agent gets 10 rewards bigger than 470 consecutively, stop the training
            # 499 is never going to be reached because of the penalized reward
            if len(rewards_list) > 10:
                stop_training = False
                for reward in rewards_list[-10:]:
                    if reward < 470:
                        break
                else:
                    stop_training = True
                if stop_training:
                    break
                
        # save the model
        saver.save(sess, "checkpoints/cartpole.ckpt")
        
        # whatch a trained agent for 5 episodes
        for episode in range(1, 6):
            state = env.reset()
            while True:
                action = agent.action(state, mode="test")
                state, reward, done, _ = env.step(action)
                env.render()
                sleep(0.05)
                if done:
                    env.close()
                    break
