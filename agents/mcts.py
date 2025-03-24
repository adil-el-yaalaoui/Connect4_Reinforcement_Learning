import numpy as np
import math
from env import Connect4env
import time
from collections import namedtuple, deque
import random
from .dqn_agent import *

class Node:
    def __init__(self, env : Connect4env, parent=None, move=None):
        self.env = env.clone()  
        self.parent = parent  
        self.move = move  
        self.wins = 0  
        self.visits = 0  
        self.children :list[Node] = []  
    
    def is_fully_expanded(self):
        return len(self.children) == len(self.env.legal_moves)

    def is_terminal(self):
        return self.env.winner != -1 or len(self.env.get_moves()) == 0 

    def best_child(self, exploration_weight=1.4):

        return max(
            self.children, 
            key=lambda child: (child.wins / (child.visits + 1e-6)) + 
                              exploration_weight * np.sqrt(np.log(self.visits + 1) / (child.visits + 1e-6))
        )
    
    def policy(self,exploration_weight=1.4):
        return [(child.wins / (child.visits + 1e-6)) + 
                              exploration_weight * np.sqrt(np.log(self.visits + 1) / (child.visits + 1e-6)) for child in self.children]





class MCTS:
    def __init__(self,nb_simulations, model : DQN, memory_buffer : ReplayMemory, epsilon : float=0.99,eps_decay:float=.9995) -> None:
        self.nb_simulations=nb_simulations
        self.model=model
        self.memory_buffer=memory_buffer
        self.epsilon=epsilon
        self.eps_decay=eps_decay

    def search(self, root_game : Connect4env):
        root = Node(root_game)
        for _ in range(self.nb_simulations):
            node = self.selection(root)
            result = self.simulation(node.env)
            self.backpropagation(node, result)

        best_child = root.best_child()

        state=root_game.board.flatten()

        action=self.choose_action_greedy(root)

        next_state=best_child.env.board.flatten()

        reward = 1 if best_child.env.winner == root_game.current_player else -1  
        winner=best_child.env.winner

        self.memory_buffer.push(state, action, next_state, reward, winner)

        self.epsilon *= self.eps_decay

        return action

    def choose_action_greedy(self,root :Node):
        #if np.random.rand() < self.epsilon:
            #return np.random.choice(root.env.legal_moves)
        #else:
        return root.best_child().move

    def selection(self, node : Node):
        while not node.is_terminal():
            if not node.is_fully_expanded():
                return self.expand(node)
            else:
                node = node.best_child(exploration_weight=2.0)
        return node

    def expand(self, node:Node):

        tried_moves = {child.move for child in node.children}
        for move in node.env.legal_moves:
            if move not in tried_moves:
                new_game = node.env.clone()
                new_game.step(move)
                new_node = Node(new_game, parent=node, move=move)
                node.children.append(new_node)
                return new_node
        return node

    def simulation(self, game : Connect4env):
       
        sim_game = game.clone()
        
        while sim_game.winner==-1:
            if sim_game.legal_moves==[]:
                break
            state_tensor=torch.tensor(sim_game.board.flatten(),dtype=torch.float32)
            with torch.no_grad():
                action_values=self.model(state_tensor)

            legal_moves = sim_game.legal_moves
            move = legal_moves[np.argmax(action_values.numpy()[legal_moves])]

            sim_game.step(move) # makes the env evolve
            
        return sim_game.winner

    def backpropagation(self, node : Node, result):
        """Met à jour les statistiques de chaque nœud en remontant."""
        while node is not None:
            node.visits += 1

            if result == node.parent.env.current_player if node.parent else node.env.current_player: 
                node.wins += 1
            elif result == 0:  # Match nul
                node.wins += 0.5 

            node = node.parent # goes back up until root node == start of the game




def train_model(model : DQN, target_model : DQN, memory : ReplayMemory ,optimizer, batch_size=64 ,gamma=0.95,step=0,target_update_freq=1000):

    if len(memory) < batch_size:

        return

    batch = memory.sample(batch_size)
    
    states, actions, next_states, rewards, winners = zip(*batch)
    
    states = torch.tensor(np.array(states), dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
    terminals = torch.tensor([winner != -1 for winner in winners], dtype=torch.bool)


    current_q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze()

    with torch.no_grad():
        next_q_values = target_model(next_states).max(1)[0]

    target_q_values = rewards + (gamma * next_q_values * (terminals))

    loss = nn.MSELoss()(current_q_values, target_q_values.detach())

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    if step % target_update_freq == 0:
        target_model.load_state_dict(model.state_dict())

    return loss.item()

