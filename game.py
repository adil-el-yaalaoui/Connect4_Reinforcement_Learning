import tkinter as tk
from tkinter import messagebox
from tkinter import *
from env import Connect4env
from agents import DQN,ReplayMemory,MCTS

import torch

environment=Connect4env()
environment.reset()

replay_memory = ReplayMemory(capacity=10_000)
model = DQN(n_observations=42,n_actions=7)
model.load_state_dict(torch.load("model_mcts_pretrained.pth"))

mcts = MCTS(nb_simulations=1000, model=model, memory_buffer=replay_memory)


window = tk.Tk()
window.title("Connect 4")

WIDTH , HEIGHT= environment.width,environment.height

canvas = tk.Canvas(window, width=WIDTH * 100, height=HEIGHT * 100,bg="white")
canvas.grid(row=1, column=0, columnspan=WIDTH+1)



map_color={1:"Red",2:"Yellow"}

grid = [[None for _ in range(WIDTH+1)] for _ in range(HEIGHT)]
boutons=[]

def jouer(colonne):
    environment.step(colonne)
    
    plot_game()
    update_boutons(environment.legal_moves)

    if environment.winner != -1:
        afficher_gagnant()
    
    window.after(100, mcts_move)  

def mcts_move():

    action = mcts.search(environment)
    

    environment.step(action)
    

    plot_game()
    update_boutons(environment.legal_moves)
    
    # Check if there is a winner
    if environment.winner != -1:
        afficher_gagnant()
    


def update_boutons(legal_moves):
    full_legal_moves=[0,1,2,3,4,5,6]
    for index in full_legal_moves:
        if index not in legal_moves:
            boutons[index].config(state=tk.DISABLED)

def reset_game():
    environment.reset()  
    canvas.delete("all")
    plot_grid()
    plot_game()
    create_columns()

def afficher_gagnant():
    message = f"Winner is: {map_color[environment.current_player]}!"
    if messagebox.askyesno("Gagnant", f"{message}\nVoulez-vous recommencer la partie ?"):
        reset_game()
    else:
        window.quit()  


def create_columns():
    for j in range(WIDTH):
        if j in environment.legal_moves:
            bouton = tk.Button(window, text="↓", width=5, height=2,command=lambda j=j: jouer(j))
            bouton.grid(row=0, column=j)
            boutons.append(bouton)

def plot_grid():

    for x in range(0, WIDTH*100, 100):
        canvas.create_line(x, 0, x, HEIGHT*100, fill="black")
    
    for y in range(0, HEIGHT*100, 100):
        canvas.create_line(0, y, WIDTH*100, y, fill="black")


def plot_game():
    for i in range(WIDTH):
        for j in range(HEIGHT):
            couleur = environment.board[i,j]
            if couleur:
                canvas.create_oval(i * 100 + 90, (HEIGHT-1-j) * 100 + 90, i * 100 + 10, (HEIGHT-1-j) * 100 + 10, fill=map_color[couleur], outline="black")
                

create_columns()

plot_grid()
plot_game()
mcts_move()
window.mainloop()