import gymnasium as gym
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import os
from PIL import Image


#Deep Q Net 
class QNetwork(nn.Module):
    def __init__(self, state_space, action_space, seed):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_space, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_space)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


#Prioritized replay buffer for more efficient sampling
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha):
        self.capacity = capacity
        self.alpha = alpha
        self.priority_array = np.zeros((capacity,), dtype=np.float32)
        self.buffer = []
        self.position = 0
        self.max_priority = 1.0

    def add(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        self.priority_array[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta):
        if len(self.buffer) == self.capacity:
            priority_array = self.priority_array
        else:
            priority_array = self.priority_array[:self.position]

        probability_array = priority_array ** self.alpha
        probability_array /= probability_array.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probability_array)
        samples = [self.buffer[idx] for idx in indices]

        weights = (len(self.buffer) * probability_array[indices]) ** (-beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            torch.from_numpy(np.vstack(states)).float().to(device),
            torch.from_numpy(np.vstack(actions)).long().to(device),
            torch.from_numpy(np.vstack(rewards)).float().to(device),
            torch.from_numpy(np.vstack(next_states)).float().to(device),
            torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(device),
            indices,
            torch.from_numpy(weights).float().to(device)
        )

    def update_priorities(self, indices, priority_array):
        for idx, priority in zip(indices, priority_array):
            self.priority_array[idx] = float(priority) 
        self.max_priority = max(self.max_priority, float(np.max(priority_array)))

    def __len__(self):
        return len(self.buffer)
    
#Agent Class
class Agent():
    def __init__(self, state_space, action_space, seed):
        self.state_space = state_space
        self.action_space = action_space
        self.seed = random.seed(seed)

        self.active_qnet = QNetwork(state_space, action_space, seed).to(device)
        self.stable_qnet = QNetwork(state_space, action_space, seed).to(device)
        self.optimizer = optim.Adam(self.active_qnet.parameters(), lr=LR)

        self.memory = PrioritizedReplayBuffer(BUFFER_SIZE, ALPHA)
        self.time_step = 0
        self.beta = BETA_START

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        self.time_step = (self.time_step + 1) % UPDATE_EVERY
        if self.time_step == 0:
            if len(self.memory) > BATCH_SIZE:
                self.learn(GAMMA)

    def act(self, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.active_qnet.eval()
        with torch.no_grad():
            action_values = self.active_qnet(state)
        self.active_qnet.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_space))

    def learn(self, gamma):
        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(BATCH_SIZE, self.beta)

        Q_targets_next = self.stable_qnet(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        Q_expected = self.active_qnet(states).gather(1, actions)

        loss = (weights * F.mse_loss(Q_expected, Q_targets, reduction='none')).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.active_qnet, self.stable_qnet, TAU)

        td_errors = torch.abs(Q_targets - Q_expected).detach().cpu().numpy().flatten()
        self.memory.update_priorities(indices, td_errors)


        self.beta = min(1.0, self.beta + (1.0 - BETA_START) / BETA_FRAMES)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


#train model
def train_model(episode_count=1500, time_steps=1000, epsilon_init=1.0, epsilon_min=0.01, epsilon_decay=0.995):
    scores = []
    recent_scores = deque(maxlen=100)
    epsilon = epsilon_init
    
    frames = []
    
    for episode in range(1, episode_count+1):
        state, _ = env.reset()
        score = 0
        for t in range(time_steps):
            action = agent.act(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            
            #one frame every 10 steps in 150 episodes
            if episode % 150 == 0 and t % 10 == 0:
                frame = env.render()
                frame = Image.fromarray(frame)
                frames.append(frame)
            
            if done:
                break 
        
        recent_scores.append(score)
        scores.append(score)
        epsilon = max(epsilon_min, epsilon_decay*epsilon)
        print(f'\rEpisode: {episode}\tAverage Score: {np.mean(recent_scores):.2f}', end="")
        if episode % 100 == 0:
            print(f'\rEpisode: {episode}\tAverage Score: {np.mean(recent_scores):.2f}')
    
    env.close()
    
    #Save gif
    #if frames:
        #frames[0].save("lunar_lander.gif", save_all=True, append_images=frames[1:], optimize=False, duration=100, loop=0)
        #print("Training GIF saved as lunar_lander.gif")
    
    return scores



#make environment
env = gym.make('LunarLander-v2', render_mode='rgb_array')

#seed standardization
env.reset(seed=0)
random.seed(0)
torch.manual_seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#parameters
BUFFER_SIZE = int(100000)
BATCH_SIZE = 64
GAMMA = 0.99
TAU = .001
LR = .0005
UPDATE_EVERY = 4
ALPHA = 0.6
BETA_START = 0.4
BETA_FRAMES = 100000

#initialize agent instance
agent = Agent(state_space=8, action_space=4, seed=0)
scores = train_model()

#plot score over episodes
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.title('Training Score vs Episode')
plt.ylabel('Score')
plt.xlabel('Episode')
#plt.savefig('training_results.png')
plt.show()

