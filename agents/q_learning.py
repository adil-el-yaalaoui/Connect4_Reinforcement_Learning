import numpy as np
from collections import defaultdict

class QLearning:
    def __init__(self,env,epsilon=0.99,eps_decay=.999,gamma=1.0,alpha=0.02) -> None:
        self.env=env
        self.last_action = None
        self.last_state = None
        self.epsilon = epsilon
        self.gamma = gamma

        self.alpha = alpha
        self.num_actions = env.action_space.n
        self.eps_decay=eps_decay

        self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))


    def reset(self):
        self.epsilon = max(self.eps_decay * self.epsilon, 0.05)
        

    def select_action(self, state,legal_moves):
        """
        The action is selected based on a epsilon-greedy
        policy using the estimate of q(S,A,W)
        """        
        self.legal_moves=legal_moves 
        flatten_state=tuple(state.flatten())   
            
        if np.random.random()>self.epsilon:
            q_values=self.q_table[flatten_state]
            legal_q_values = {a: q_values[a] for a in legal_moves}
            chosen_action=max(legal_q_values, key=legal_q_values.get)
        else:
            chosen_action=np.random.choice(legal_moves)
            
        return chosen_action

    def step(self, action ,next_state, reward, winner):

        next_state_flatten = tuple(next_state.flatten()) # S'

        # We compute the TD error with the next state and action and the previous ones
        max_action_q_learning = 0 if winner!=-1 else max(self.q_table[next_state_flatten]) # Q-Learning Step

        td_error = reward + self.gamma * max_action_q_learning - self.q_table[self.last_state][self.last_action]
        

        self.q_table[self.last_state][self.last_action] += self.alpha * td_error

        self.last_action = action
        self.last_state = next_state_flatten

        

    
