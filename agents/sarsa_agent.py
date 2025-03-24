import numpy as np
from collections import defaultdict

class SarsaLambda:
    def __init__(self,env,epsilon=0.99,eps_decay=.999,gamma=1.0,alpha=0.02,lambd=0.98) -> None:
        self.env=env
        self.last_action = None
        self.last_state = None
        self.epsilon = epsilon
        self.gamma = gamma

        self.alpha = alpha
        self.num_actions = env.action_space.n
        self.previous_tiles = None
        self.eps_decay=eps_decay
        self.lambd=lambd

        self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))

        self.traces = defaultdict(lambda: np.zeros(env.action_space.n))


    def reset(self):
        self.traces.clear()
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

    def step(self, next_state, reward, winner):

        next_state_flatten = tuple(next_state.flatten())

        action = self.select_action(next_state,self.legal_moves)

        # We compute the TD error with the next state and action and the previous ones
        next_value = 0 if winner!=-1 else self.q_table[next_state_flatten][action]

        td_error = reward + self.gamma * next_value - self.q_table[self.last_state][self.last_action]
        
        self.traces[self.last_action][self.previous_tiles] = 1 # Replacing traces

        self.q_table[self.last_state][self.last_action] += self.alpha * td_error * self.traces[self.last_state][self.last_action] # We update the weights

        # Decay traces if not done
        if winner==-1:
            self.traces[self.last_state][self.last_action] *= self.gamma * self.lambd


        self.last_action = action
        self.last_state = next_state_flatten

        

    
