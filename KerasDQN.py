from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from time import sleep
import numpy as np
import random
import gym


class DQNAgent:
    """Class providing DQN algorithm based on keras"""
    def __init__(self, state_size, action_size, gamma=1.0, epsilon_start=1.0, epsilon_decay=0.0001,
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
        self.model = self._build_model(n_Hlayers, n_nodes)
        self.step = 0
        
    def _build_model(self, hidden_layers, nodes):
        """Build the neural network structure"""
        model = Sequential()
        model.add(Dense(nodes, input_dim=self.state_size, activation='relu'))
        for layer in range(hidden_layers):
            model.add(Dense(nodes, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer=Adam(lr=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        """collect past experience"""
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state, mode='train'):
        """Return an action based on the current state, if mode is test the agent choose
        the best action that gives the biggest reward.
        If mode is train, then it can choose also a random action with epsilon probability"""
        self.step += 1
        # reduce gradually epsilon to its minimum value
        self.epsilon = self.epsilon_min + (
            self.epsilon_max - self.epsilon_min)*np.exp(-self.epsilon_decay*self.step)
        if np.random.rand() > self.epsilon or mode.lower() == "test":
            return np.argmax(self.model.predict(state)[0])
        else:
            return random.randrange(self.action_size)

    def save(self, path="model.h5"):
        self.model.save_weights(path)

    def load(self, path="model.h5"):
        self.model.load_weights(path)
    
    def replay(self, batch_size):
        """Train the agent with a random batch of the collected data"""
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
              target = reward + self.gamma * \
                       np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)


def collect_experience(env_, agent_, size):
    """Save data in agent's memory, preparing it for training"""
    env_.reset()
    state, reward, done, _ = env_.step(env_.action_space.sample())
    for data in range(size):
        action = env_.action_space.sample()
        next_state, reward, done, _ = env_.step(action)
        # penalize reward based on the position of the cart
        reward = reward * (1 - abs(next_state[1]/2.4))
        # reshape states
        state = np.reshape(state, [1, 4])
        next_state = np.reshape(next_state, [1, 4])
        # save experience in agent's memory
        agent_.remember(state, action, reward, next_state, done)
        if done:
            env_.reset()
            state, reward, done, _ = env_.step(env.action_space.sample())
        else:
            state = next_state

            
if __name__ == "__main__":
    # initialize gym environment and the agent
    env = gym.make('CartPole-v1')
    agent = DQNAgent(4, 2)
    episodes = 700
    batch_size = 32
    
    # populate memory
    collect_experience(env, agent, batch_size)
    
    # Iterate the game
    rewards_list = list()
    max_steps = 500
    for e in range(episodes):
        # reset state in the beginning of each game
        state = env.reset()

        # change state shape
        state = np.reshape(state, [1, 4])
        
        score=0
        for time_t in range(max_steps):
            # Take action, get new state and reward
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            
            # adjust reward, gets bigger reward if cart stays in the center
            reward = reward * (1 - abs(next_state[1]/2.4))

            # collect total reward
            score += reward

            # change state shape
            next_state = np.reshape(next_state, [1, 4])
            
            # Remember the previous state, action, reward, and done
            agent.remember(state, action, reward, next_state, done)
            
            # make next_state the new current state for the next frame.
            state = next_state

            # if episode ends
            if done:
                print("episode: {}/{}, score: {:.2f}, e: {:.4f}"
                      .format(e, episodes, score, agent.epsilon))
                
                # Add reward to list
                rewards_list.append(score)
                break
            
            # train the agent with the experience of the episode
            agent.replay(batch_size)
            
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
    agent.save("checkpoints/cartpole.h5")

    # whatch a trained agent for 5 episodes
    del agent
    agent = DQNAgent(4, 2)
    agent.load("checkpoints/cartpole.h5")
    for e in range(1, 6):
        state = env.reset()
        while True:
            action = agent.act(np.reshape(state, [1, 4]), "test")
            next_state, reward, done, _ = env.step(action)
            env.render()
            sleep(0.05)
            state = np.reshape(next_state, [1, 4])
            if done:
                env.close()
                break
