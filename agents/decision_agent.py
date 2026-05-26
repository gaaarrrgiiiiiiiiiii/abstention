import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from agents.gating_network import GatingNetwork
from agents.reward import calculate_reward

class DecisionAgent:
    """
    The Decision Agent uses Reinforcement Learning (REINFORCE algorithm)
    to optimize the abstention policy. It wraps the GatingNetwork.
    """
    
    def __init__(self, feature_dim, uncertainty_dim=3, lr=1e-3, device='cpu'):
        self.device = device
        self.policy_net = GatingNetwork(feature_dim, uncertainty_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Memory for REINFORCE
        self.saved_log_probs = []
        self.rewards = []
        
    def select_action(self, features, uncertainties):
        """
        Selects an action (0: Don't Abstain, 1: Abstain) based on the policy.
        """
        abstain_prob = self.policy_net(features, uncertainties)
        
        # We model this as a Bernoulli distribution
        # abstain_prob is the probability of taking action 1 (Abstain)
        probs = torch.cat([1 - abstain_prob, abstain_prob], dim=1)
        m = Categorical(probs)
        action = m.sample()
        
        self.saved_log_probs.append(m.log_prob(action))
        
        return action.item()
        
    def store_reward(self, reward):
        self.rewards.append(reward)
        
    def update_policy(self, gamma=0.99):
        """
        Performs a REINFORCE update step.
        """
        if not self.saved_log_probs:
            return
            
        R = 0
        policy_loss = []
        returns = []
        
        # Calculate discounted returns
        for r in self.rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns).to(self.device)
        # Normalize returns for stability
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
            
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        
        # Clear memory
        del self.saved_log_probs[:]
        del self.rewards[:]
        
        return policy_loss.item()
