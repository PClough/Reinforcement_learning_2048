# -*- coding: utf-8 -*-
"""
written by PClough
based on 
https://medium.com/@jofre44/game-app-with-python-and-tkinter-let-s-play-2048-e9e25223a711

using Tensorforce reinforcement learning
https://github.com/tensorforce/tensorforce/blob/master/examples/carla_examples.py

This code will use reinforcement learning to determine 
the correct keys to press to produce 2048 / win the game

"""

#%% 
import tkinter as tk
import random
#import time
import os
from datetime import datetime
from pynput.keyboard import Key, Controller
import matplotlib.pyplot as plt
#import numpy as np
# import math
from tensorforce.environments import Environment
from tensorforce.agents import Agent


# Controller for automating key presses
kb = Controller()

window = tk.Tk()

# Design parameters, color in Hex
GRID_COLOR = "#a6bdbb"
EMPTY_CELL_COLOR = "#c2b3a9"
SCORE_LABEL_FONT = ("Verdana", 18)
SCORE_FONT = ("Helvetica", 24, "bold")
CELL_COLORS = {2: "#fcefe6", 4: "#f2f8cb", 8: "#f5b682", 16: "#f29446", 32: "#ff775c", 64: "#e64c2e", 128: "#ede291", 
               256: "#fce130", 512: "#ffdb4a", 1024: "#f0b922", 2048: "#fad74d", 4096: '#249a91'}
CELL_NUMBER_COLORS = {2: "#695c57", 4: "#695c57", 8: "#ffffff"}
CELL_NUMBER_FONTS = ("Helvetica", 15, "bold")


#%% Environment definition
class My_2048_Environment(Environment, tk.Frame):
    """This class defines the learning environment.
    """
    def __init__(self):
        ## Some initializations.
        
        # Set main window
        window.lift()
        window.focus_force()
        window.attributes("-topmost", True)
        window.after_idle(window.attributes,'-topmost',False)
        tk.Frame.__init__(self)
        self.grid()
        self.master.title("2048")
        self.main_grid = tk.Frame(self, bg=GRID_COLOR, bd=3, width=100, height=100)
        self.main_grid.grid(pady=(100,0))
        # Game functions and parameters
        # Top value to play the game
        self.top_value = 2048
        # Grid size
        self.grid_size = 5
        # Main window position
        self.sw = self.master.winfo_screenwidth()
        self.sh = self.master.winfo_screenheight()
        # Game initialization
        self.make_GUI()
        self.create_button()
        self.start_game()
        self.timestep = 0
        self.score = 0
        self.prev_score = 0
        self.current_score = 0
        
        # Defining buttons to play 
        self.master.bind("<Left>", self.left)
        self.master.bind("<Right>", self.right)
        self.master.bind("<Up>", self.up)
        self.master.bind("<Down>", self.down)
    
    # Functions to set game desing
    def make_GUI(self):
        self.cells = []
        # Creating the grid 
        for i in range(self.grid_size):
            row = []
            for j in range(self.grid_size):
                cell_frame = tk.Frame(self.main_grid, bg=EMPTY_CELL_COLOR, width=80, height=80)
                cell_frame.grid(row=i, column=j, padx=5, pady=5)
                cell_number = tk.Label(self.main_grid, bg=EMPTY_CELL_COLOR)
                cell_number.grid(row=i, column=j)
                cell_data = {'frame': cell_frame, "number": cell_number}
                row.append(cell_data)
            self.cells.append(row)
        # Game position in screen
        w = self.grid_size*91
        h = (self.grid_size+1)*93
        x = (self.sw - w)/2
        y = (self.sh - h)/2
        self.master.geometry('%dx%d+%d+%d' % (w, h, x, y))  
        # Game title
        act_frame = tk.Frame(self)
        act_frame.place(relx=0.10, rely=0.05, anchor="center",)
        tk.Label(
            act_frame,
            text="2048",
            font=SCORE_LABEL_FONT,
        ).grid(row=0)  
        # Game current score and best score
        self.score = 0
        self.bstScore = 0
        if os.path.exists("bestscore.ini"):
            with open("bestscore.ini", "r") as f:
                self.bstScore = int(f.read())    
        score_frame = tk.Frame(self)
        score_frame.place(relx=0.5, y=45, anchor="center")
        tk.Label(score_frame, text="Score", font=SCORE_LABEL_FONT).grid(row=0)
        self.score_label = tk.Label(score_frame, text=self.score, font=SCORE_FONT)
        self.score_label.grid(row=1)
        record_frame = tk.Frame(self)
        record_frame.place(relx=0.8, y=45, anchor="center")
        tk.Label(record_frame, text="Record", font=SCORE_LABEL_FONT).grid(row=0)
        self.record_label = tk.Label(record_frame, text= self.bstScore, font=SCORE_FONT)
        self.record_label.grid(row=2)

    # Button for game restart 
    def create_button(self):
        button = tk.Button(self, text='New Game', command=lambda: self.new_game())
        button.place(relx=0.1, rely=0.10, anchor="center")
        
    # Function for game restart
    def new_game(self):
        self.make_GUI()
        self.start_game()
        
    # Creation of new game
    def start_game(self):
        # Place the first number in a random position
        self.matrix = [[0]*self.grid_size for _ in range(self.grid_size)]
        row = random.randint(0, self.grid_size-1)
        col = random.randint(0, self.grid_size-1)
        self.matrix[row][col] = 2
        self.cells[row][col]["frame"].configure(bg=CELL_COLORS[2])
        self.cells[row][col]["number"].configure(
            bg=CELL_COLORS[2],
            fg=CELL_NUMBER_COLORS[2],
            font=CELL_NUMBER_FONTS,
            text="2"
        )
        # Place the second number in an empty random position
        while(self.matrix[row][col] !=0):
            row = random.randint(0, self.grid_size-1)
            col = random.randint(0, self.grid_size-1)
        self.matrix[row][col] = 2
        self.cells[row][col]["frame"].configure(bg=CELL_COLORS[2])
        self.cells[row][col]["number"].configure(
            bg=CELL_COLORS[2],
            fg=CELL_NUMBER_COLORS[2],
            font=CELL_NUMBER_FONTS,
            text="2"
        )
        self.old_matrix = self.matrix
        self.score = 0
        self.prev_score = 0
        self.current_score = 0
        self.timestep = 0

    # Stack number 
    def stack(self):
        new_matrix = [[0] * self.grid_size for _ in range(self.grid_size)]
        for row in range(self.grid_size):
            fill_position = 0
            for col in range(self.grid_size):
                if self.matrix[row][col] != 0:
                    new_matrix[row][fill_position] = self.matrix[row][col]
                    fill_position += 1
        self.matrix = new_matrix

    # Combine equal numbers
    def combine(self):
        for row in range(self.grid_size):
            for col in range(self.grid_size-1):
                if (self.matrix[row][col] != 0) and (self.matrix[row][col] == self.matrix[row][col + 1]):
                    self.matrix[row][col] *= 2
                    self.matrix[row][col + 1] = 0
                    self.score += self.matrix[row][col]
                    if self.score > self.bstScore:
                        self.bstScore = self.score
                        print("\n    Beaten the best score!!!   \n")
                        with open("bestscore.ini", "w") as f:
                            f.write(str(self.bstScore))

    # Reverse function    
    def reverse(self):
        new_matrix = []
        for row in range(self.grid_size):
            new_matrix.append([])
            for col in range(self.grid_size):
                new_matrix[row].append(self.matrix[row][(self.grid_size-1) - col])
        self.matrix = new_matrix

    # Transpose function
    def transpose(self):
        new_matrix = [[0]*self.grid_size for _ in range(self.grid_size)]
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                new_matrix[row][col] = self.matrix[col][row]
        self.matrix = new_matrix

    # Add new number in a random position
    def add_new_tile(self):
        if any(0 in row for row in self.matrix):
            row = random.randint(0,self.grid_size-1)
            col = random.randint(0,self.grid_size-1)
            while(self.matrix[row][col] != 0):
                row = random.randint(0,self.grid_size-1)
                col = random.randint(0,self.grid_size-1)
            self.matrix[row][col] = random.choice([2, 4])

    # Functions to update de GUI
    def update_GUI(self):
        cell_text_color = 0
        cell_cell_color = 0
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                cell_value = self.matrix[row][col]
                if cell_value == 0:
                    self.cells[row][col]["frame"].configure(bg=EMPTY_CELL_COLOR)
                    self.cells[row][col]["number"].configure(bg=EMPTY_CELL_COLOR, text="")
                else:
                    if cell_value >= 8:
                        cell_text_color = 8
                    else:
                        cell_text_color = cell_value
                    if cell_value >= 4096:
                        cell_cell_color = 4096
                    else:
                        cell_cell_color = cell_value
                    
                    self.cells[row][col]["frame"].configure(bg=CELL_COLORS[cell_cell_color])
                    self.cells[row][col]["number"].configure(
                        bg=CELL_COLORS[cell_cell_color], 
                        fg=CELL_NUMBER_COLORS[cell_text_color],
                        font=CELL_NUMBER_FONTS,
                        text=str(cell_value))
        self.score_label.configure(text=self.score)
        self.record_label.configure(text=self.bstScore)
        self.update_idletasks()

    # Check for possibles moves
    def any_move(self):
        for i in range(self.grid_size):
            for j in range(self.grid_size-1):
                if self.matrix[i][j] == self.matrix[i][j + 1] or \
                   self.matrix[j][i] == self.matrix[j + 1][i] :
                    return True
        return False

    # Check for game over
    # def game_over(self):
    #     # Check if tovalue is reached
    #     # Check if there are no more moves in the grid
    #     if not any(0 in row for row in self.matrix) and not self.any_move():
    #         #self.popup("Game Over!!", "Game Over!!")
    #         #self.quit()
    #         # self.destroy()
    #         terminal = True
            
            
    # Left stacking
    def left(self, event):
        self.stack()
        self.combine()
        self.stack()
        self.add_new_tile()
        self.update_GUI()
        #self.game_over()

    # Right stacking
    def right(self, event):
        self.reverse()
        self.stack()
        self.combine()
        self.stack()
        self.reverse()
        self.add_new_tile()
        self.update_GUI()
        #self.game_over()

    # Up stacking
    def up(self, event):
        self.transpose()
        self.stack()
        self.combine()
        self.stack()
        self.transpose()
        self.add_new_tile()
        self.update_GUI()
        #self.game_over()

    # Down stacking
    def down(self, event):
        self.transpose()
        self.reverse()
        self.stack()
        self.combine()
        self.stack()
        self.reverse()
        self.transpose()
        self.add_new_tile()
        self.update_GUI()
        #self.game_over()
        
    def seed(self, seed):
        """
        Sets the random seed of the environment to the given value (current time, if seed=None).
        Naturally deterministic Environments don't have to implement this method.

        Args:
            seed (int): The seed to use for initializing the pseudo-random number generator (default=epoch time in sec).
        Returns: The actual seed (int) used OR None if Environment did not override this method (no seeding supported).
        """
        return None
        
    def states(self):
        # returns the matrix
        return dict(type='float', shape=(self.grid_size, self.grid_size))

    def actions(self):
        return dict(type='int', num_values=4)
    
    def reset(self):
        """Reset state.
        """
        self.new_game()
        return self.matrix

    def execute(self, actions):
        ## Check the action is either L, R, U, D.
        # Action 1 means left button press, 2 = right, 3 = up, 4 = down
        assert actions == 0 or actions == 1 or actions == 2 or actions == 3
        #print("actions: " + str(actions))
        if actions == 0:
            #print("here = 0")
            kb.press(Key.left)
            kb.release(Key.left)
            # print("Pressed left")
        elif actions == 1:
            #print("here = 1")
            kb.press(Key.right)
            kb.release(Key.right)
            # print("Pressed right")
        elif actions == 2:
            #print("here = 2")
            kb.press(Key.down)
            kb.release(Key.down)
            # print("Pressed down")
        elif actions == 3:
            #print("here = 3")
            kb.press(Key.up)
            kb.release(Key.up)
            # print("Pressed up")
        
        # REWARDS
        a = 0
        for i in range(0, self.grid_size):
            for j in range(0, self.grid_size):
                if self.old_matrix[i][j] == self.matrix[i][j]:
                    a += 1
        if a == 25:
            self.old_matrix = self.matrix           
        
        ## Update the current_temp
        self.current_score = self.score # self.response(actions)
        
        ## Compute the reward 
        
        # reward for getting closer or beyond 2048
        reward_2048 = (1 - (1 - (max(max(self.matrix))/2048) )) *10
        # reward_2048 = 1
        
        # reward for having high value numbers in the cells
        reward_matrix_sum = sum([sum(i) for i in zip(*self.matrix)])/100
        
        # reward for high score
        # reward_score = self.current_score/10000
        reward_score = 1
        
        # Reward for increasing the score
        # if self.current_score > self.prev_score:
        #     reward_inc = 10
        # else:
        #     reward_inc = 1
        reward_inc = 1
        self.prev_score = self.current_score
        
        # Negative reward for not making a decision when there are moves to make
        if self.timestep > 1 and self.any_move():
            neg_reward_timestep = -1 #*self.timestep
            # Multiply all rewards together
            reward = neg_reward_timestep
        else:
            neg_reward_timestep = 1
            # Multiply all rewards together
            reward = reward_inc * reward_2048 * neg_reward_timestep * reward_matrix_sum * reward_score
                
        # print("anymove? " + str(self.any_move()))
        # print("reward_inc = " + str(reward_inc))
        # print("reward_2048 = " + str(reward_2048))
        # print("neg_reward_timestep = " + str(neg_reward_timestep))
        # print("reward_matrix_sum = " + str(reward_matrix_sum))
        # print("reward_score = " + str(reward_score))
        # print("timestep = " + str(self.timestep))
        # print("Reward = " + str(reward) + "\n")
        
        # terminal == False means episode is not done
        # terminal == True means it is done.
        # Check if game ended
        if not any(0 in row for row in self.matrix) and not self.any_move():
            terminal = True
        else:
            terminal = False
        
        ## Increment timestamp if there has been no change
        if self.old_matrix == self.matrix:
            self.timestep += 1
        else:
            if self.timestep > 0:
                reward = reward*1.5 # Give it a big reward for trying a move after it gets stuck
            self.timestep = 0
        
        if self.timestep > 10:
            terminal = True
        
        self.old_matrix = self.matrix 
        
        return self.matrix, terminal, reward


#%% Create the environment
###   - Tell it the environment class
###   - Set the max timestamps that can happen per episode
environment = Environment.create(environment=My_2048_Environment) #, max_episode_timesteps=500)
environment.seed(0)

agent = Agent.create(
    agent='tensorforce', environment=environment, update=1, exploration = 10.0, 
    optimizer=dict(optimizer='adam', learning_rate=1e-3, multi_step=10, 
                   linesearch_iterations=5), #, doublecheck_update=True),
    objective='policy_gradient', reward_estimation=dict(horizon=1), 
    actions = dict(type='int', num_values=4)
)


#%% Training episodes

score_recording_training = list()
episode_recording = list()
t = 0

plt.figure()
plt.xlabel('Timestep')
plt.ylabel('Reward')
plt.title('Training')

Num_of_episodes = 100
for episode_train in range(Num_of_episodes):
    states = environment.reset()
    terminal = False
    print("\nEpisode training # : " + str(episode_train+1)) # because 0th order indexing is not my favourite. I <3 MATLAB.
    #time.sleep(0.1)
    episode_recording.append(t)
    
    while not terminal:
        #time.sleep(0.01)
        window.update()
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)
        score_recording_training.append(reward)
        t += 1
    
    print("highest number: " + str(max(max(states))))
    
    plt.plot(range(len(score_recording_training)), score_recording_training, "-k", label = "score recording")
    # plt.plot(episode_recording, [range(0,100)], "--r") # , label = "new episodes") [ele for ele in [range(0,100)] for _ in range(Num_of_episodes)]
    for xc in episode_recording:
        plt.axvline(x = xc, linestyle = '--', color = 'r')
    plt.legend() 
    plt.draw()
    plt.show()

           
#%% Save agent
# datetime object containing current date and time
now = datetime.now()

# dd/mm/YY H:M:S
dt_string = now.strftime("%Y-%m-%d %H-%M-%S")
#print("date and time =", dt_string)

Agent_saved = agent.save(directory="C:/Users/e804491/Dropbox/Python Scripts/", filename = dt_string, format='checkpoint', append='episodes')
print("Agent Saved: " + str(Agent_saved))

#%% load saved agent

# agent = Agent.load(directory='C:/Users/e804491/Dropbox/Python Scripts/', 
#                           filename = '2023-11-17 11-56-02-172-1',
#                           format='checkpoint', environment=environment, 
#                           objective='policy_gradient', 
#                           reward_estimation=dict(horizon=1),
#                           optimizer=dict(optimizer='adam', learning_rate=1e-3, multi_step=10, linesearch_iterations=5),
#                           update = 1)


#%% Test the models performance 

states = environment.reset()

# environment.current_score = 0
# states = environment.current_score

internals = agent.initial_internals()
terminal = False

print("\nTesting the performance")

### Run an episode
score_recording = list()
while not terminal:
    window.update()
    actions, internals = agent.act(states=states, internals=internals, independent=True, deterministic = False)
    states, terminal, reward = environment.execute(actions=actions)
    score_recording.append(reward)



#%% Plot the run
plt.figure()
plt.plot(range(len(score_recording)), score_recording)
plt.xlabel('Timestep')
plt.ylabel('Reward')
now = datetime.now()
plt.title('Testing : ' + now.strftime("%Y-%m-%d %H-%M-%S"))
plt.show()        















#%% END
