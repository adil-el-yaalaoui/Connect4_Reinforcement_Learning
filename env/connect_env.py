from typing import  Optional
import numpy as np
import gym
from gym.spaces import Box, Discrete, Tuple

class Connect4env(gym.Env):
    """
    Board :
        0 for empty
        1 for player 1
        2 for player 2
    
    """
    def __init__(self,width=7,height=6,nb_of_connect=4) -> None:

        self.width=width
        self.height=height
        self.connect=nb_of_connect

        # Each player has its own space which is the game space
        player_observation_space=Box(low=0,
                                   high=1,
                                   shape=(self.width, self.height),
                                   dtype=np.int32)
        
        self.observation_space = player_observation_space

        self.action_space = Discrete(self.width)
        self.reset()
    
    def reset(self) :
        self.board = np.full((self.width, self.height), 0)
        self.winner = -1

        self.current_player = np.random.choice([1,2])
        self.ennemy=self.current_player%2 +1
        self.legal_moves=self.get_moves()
        return self.get_obs(),self.legal_moves
    
    def switch_players(self):
        self.current_player=2 if self.current_player==1 else 1
        self.ennemy=self.current_player%2 +1

    def get_obs(self):
        return self.board
    
    def make_move(self, movecol):
        """

        Function that fills the board with the move from the player 
        and a random move from the other player for now

        """
  
        if not(movecol >= 0 and movecol <= self.width and self.board[movecol][self.height - 1] == 0):
            raise ValueError("This is not a valid move, the column is full or you are out of bounds")
        
        row=self.height-1

        while row >=0 and self.board[movecol][row]==0:
            row-=1
        
        # Now we are at the row with a free spot
        row+=1
        self.board[movecol][row]=self.current_player
        self.legal_moves=self.get_moves()

        return movecol,row


    def step(self, movecol):
        """
        Perform a step for the env after the column that has be chosen to play
        """

        _,row_player=self.make_move(movecol)

        reward_vector={1:0 , 2:0}

        if self.does_move_win(movecol, row_player):
            self.winner=self.current_player

            reward_vector[self.current_player]=1
            reward_vector[self.ennemy]=-1

            return self.get_obs(),reward_vector,self.legal_moves,self.winner
        
        elif self.legal_moves==[]:
            self.winner=0
            return self.get_obs(), reward_vector, self.legal_moves,self.winner

        else :

            self.switch_players()
            return self.get_obs(), reward_vector, self.legal_moves,-1
                

    def clone(self):
        """
        Creates a deep copy of the game state.
        NOTE: it is _really_ important that a copy is used during simulations
              Because otherwise MCTS would be operating on the real game board.
        :returns: deep copy of this GameState
        """
        st = Connect4env(width=self.width, height=self.height)
        st.winner = self.winner
        st.board = np.array([self.board[col][:] for col in range(self.width)])
        st.legal_moves=self.legal_moves
        st.current_player=self.current_player
        return st
        
    def get_moves(self):
        """
        :returns: array with all possible moves, index of columns which aren't full
        """
        if self.winner !=-1:
            return []
        return [col for col in range(self.width) if self.board[col][self.height - 1] == 0]

    def does_move_win(self, x, y):
        """
        Checks whether a newly dropped chip at position param x, param y
        wins the game.
        :param x: column index
        :param y: row index
        :returns: (boolean) True if the previous move has won the game
        """
        me = self.board[x][y]
        for (dx, dy) in [(0, +1), (+1, +1), (+1, 0), (+1, -1)]:
            p = 1
            while self.is_on_board(x+p*dx, y+p*dy) and self.board[x+p*dx][y+p*dy] == me:
                p += 1
            n = 1
            while self.is_on_board(x-n*dx, y-n*dy) and self.board[x-n*dx][y-n*dy] == me:
                n += 1

            if p + n >= (self.connect + 1): # want (p-1) + (n-1) + 1 >= 4, or more simply p + n >- 5
                return True

        return False

    def is_on_board(self, x, y):
        return x >= 0 and x < self.width and y >= 0 and y < self.height

    def get_result(self, player):
        """
        :param player: (int) player which we want to see if he / she is a winner
        :returns: winner from the perspective of the param player
        """
        if self.winner == None: return 0  # A draw occurred
        return +1 if player == 1 else -1