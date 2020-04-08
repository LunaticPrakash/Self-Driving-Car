#importing libraries
import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable


#creating the architecture of Neural Network

class Network(nn.Module):
    
    #input_size = no. of input neuron(no. of dimension of our input)
    #nb_action = no. of output neuron(no. of actions cars can make i.e 
    #corresponds to action like go left, right, forward)
    def __init__(self, input_size, nb_action):
        super(Network,self).__init__() #to use all tools of nn.Module
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(in_features= input_size, out_features=60, bias=True) #to make full connection between neuron
        #of input layer with neuron of hidden layer
        #in_features= no.of neurons of input layer you want to connect with hidden layer
        #out_features= no.of neurons of hidden layer you want to connect with output layer(totally experimental)
        
        self.fc2 = nn.Linear(in_features=60, out_features=nb_action, bias=True) #to make full connection between neuron
        #of hidden layer with neuron of output layer
         #in_features= no.of neurons of hidden layer you want to connect with output layer
        #out_features= no.of neurons of output layer you want to connect with output layer(totally experimental)
        
    #So, now we have 5 input neurons, 60 hidden neurons,and 3 output neurons    
    
    
    #function to activate neurons and return the q_values for each state
    #like forward, left, right.
    def forward(self, state):
        #x = hidden neurons
        #Now we have to activate hidden neurons, for this first we
        #get hidden neurons from fc1 and then activate them using an
        #activation function, relu() which is "Rectifier func." itself
        x = F.relu(self.fc1(state)) #hidden neurons activated
        #now we have to return output neurons
        q_values = self.fc2(x)
        return q_values
    
#implementing experience replay
class ReplayMemory(object):
    
    def __init__(self, capacity):
        #capacity = no. of previous events we want to store
        #memory = list conating last "capacity" no. of transitions
        self.capacity= capacity
        self.memory = []
        
    def push(self, event):
        #push func. to append new transition or event in memory and make
        #sure that it has always "capacity" no. of events 
        #event consist of last-state, new-state,  last-action, last-reward
        self.memory.append(event)
        if(len(self.memory) > self.capacity):
            del self.memory[0]
            
    def sample(self,batch_size):     
        #to get random samples from previously stored transitions in memory
        #batch_size = no. of samples to be get from memory
        samples = zip(*random.sample(self.memory, batch_size))
        #it will select "batch_size" no. of random samples from memory
        #and zip fn. will reshape them into 3 samples of state, action
        #and reward like (s1,s2), (a1,a2), (r1,r2);
        #we can not return "samples" as simple, we have to convert it into Torch Variable
        return map(lambda x: Variable(torch.cat(x,0)), samples)
        
#implementing Deep Q-Learning Model

class Dqn():
    
    def __init__(self, input_size, nb_action, gamma):
        #gamma = delay coefficient
        #rewardwindow = sliding window of mean of last 100 rewards
        #just to evaluate the evolution of our ai model
        #model = our neural network
        #memory = our memory
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(capacity = 100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action =  0
        self.last_reward = 0
        
        
    def select_action(self, state):
        #fn. that will select right action at each time
        #SoftMax helps in  getting the best action to play while 
        #still exploring the other actions
        #so we input q-values that we get from our neural network
        #to softmax() and get their probabilities
        probs = F.softmax(self.model(Variable(state,volatile = True))*100) #T=100
        action = probs.multinomial(num_samples=1) #random draw from the distribution
        return action.data[0,0]
    
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        td_loss.backward(retain_graph = True)
        self.optimizer.step()
    
    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action
    
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.)
    
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, 'last_brain.pth')
    
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")
        
        
        
    
        
        
        
