#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 19:26:37 2018

@author: sarahperrin

"""

#! /usr/bin/env python3
from itertools import groupby, chain

#from keras.utils import plot_model

import numpy as np
np.set_printoptions(suppress=True)

from shutil import copyfile

from train.game import GameState
from train.agent import Agent

from train.model import Residual_CNN

from train.settings import run_archive_folder

import train.config as config

import other_agent.minimax as minimax

NONE = '.'
RED = 'R'
YELLOW = 'J'

class player :
    def __init__ (self, name, runNumber = 3, modelVersion = 29) :
        self.name = name
        self.ai_player = None
        if name == 'AI' :
            self.ai_player = createAIPlayer(runNumber,modelVersion)
            
class newGame :
    def __init__ (self, redPlayer, yellowPlayer):
        """Create a new game."""
        self.cols = 7
        self.rows = 6
        self.win = 4
        self.board = [[NONE] * 6 for _ in range(7)]
        self.redPlayer = redPlayer
        self.yellowPlayer = yellowPlayer
    
    def insert (self, column, color, display):
        """Insert the color in the given column."""
        c = self.board[column]
        if c[0] != NONE:
            raise Exception('Column is full')

        i = -1
        while c[i] != NONE:
            i -= 1
        c[i] = color
        if(display) :
            self.printBoard()
    
    def getPlayer(self,color) :
        if color == RED : 
            return self.redPlayer
        elif color == YELLOW :
            return self.yellowPlayer
        return None

    def getFirstColor(self) :
        """chose starting color randomly"""
        return np.random.choice([RED,YELLOW],1)[0]
            
    def boardIsFull(self) :
        """return True when the board is full"""
        res = True
        for i in range(self.cols) :
            res = res and self.board[i][0] != NONE
        return res
        

    def getWinner (self):
        """Get the winner on the current board."""
        lines = (
            self.board, # columns
            zip(*self.board), # rows
            diagonalsPos(self.board, self.cols, self.rows), # positive diagonals
            diagonalsNeg(self.board, self.cols, self.rows) # negative diagonals
        )

        for line in chain(*lines):
            for color, group in groupby(line):
                if color != NONE and len(list(group)) >= self.win:
                    return color
                
        return NONE

    def printBoard (self):
        """Print the board."""
        print('  '.join(map(str, range(self.cols))))
        for y in range(self.rows):
            print('  '.join(str(self.board[x][y]) for x in range(self.cols)))
        print()
        
    def getOpponent(self,color) :
        """returns the player's opponent"""
        if color==RED :
            return YELLOW
        else :
            return RED
        
    def toBoardGs(self,player) :
        """converts a board to the required format for GameState from the point of view of player"""
        board_gs = np.zeros((6,7)) #convert board to np.array 
        
        opponent = self.getOpponent(player) 
        
        # convert board to the right input for GameState
        for i in range(7):
            for j in range(6):
                if self.board[i][j]== player:
                    board_gs[j][i] = 1
                elif self.board[i][j]== opponent:
                    board_gs[j][i] = -1
                    
        return board_gs
    
    def getMove(self,current_color) :
        
        player = self.getPlayer(current_color) #player that we are trying to
                                                        #play the next move
                                                        
        if player.name == 'Random' :
            action = np.random.randint(0,7)
            while self.board[action][0] != '.' :
                action = np.random.randint(0,7)
            return action
        
        elif player.name == 'Minimax' :
            minmax = minimax.Minimax(self.board)
            res = minmax.bestMove(3, self.board, current_color)
            return res[0]
        
        elif player.name == 'AI' :
        
            board_gs = self.toBoardGs(current_color)
            
            gs = GameState(board_gs.flatten(), 1)
            
            pred = player.ai_player.get_preds(gs)[1] #Contains array of winning probability 
                                            #for each possible action
            #print(pred)
            
            action = np.argmax(pred)%7 #The future action is the one with the best 
                                        #probability of win
            return action
        
        elif player.name == 'Human' :
            action = int(input('{}\'s turn: '.format('Red' if current_color == RED else 'Yellow')))
            return action


def diagonalsPos (matrix, cols, rows):
    """Get positive diagonals, going from bottom-left to top-right."""
    for di in ([(j, i - j) for j in range(cols)] for i in range(cols + rows -1)):
        yield [matrix[i][j] for i, j in di if i >= 0 and j >= 0 and i < cols and j < rows]

def diagonalsNeg (matrix, cols, rows):
    """Get negative diagonals, going from top-left to bottom-right."""
    for di in ([(j, i - cols + j + 1) for j in range(cols)] for i in range(cols + rows - 1)):
        yield [matrix[i][j] for i, j in di if i >= 0 and j >= 0 and i < cols and j < rows]


#Function to load the NN and create the AI agent
def createAIPlayer(runNumber = 3, modelVersion = 29) :
    
    #copy the config file to root
    copyfile(run_archive_folder + 'connect4' + '/run' + str(runNumber).zfill(4) + '/config.py', './config.py')
    
    ######## LOAD MODEL ########
    best_NN = Residual_CNN(config.REG_CONST, config.LEARNING_RATE,
                           (2,) + (6,7),   42, config.HIDDEN_CNN_LAYERS)
    
    best_player_version = modelVersion
    
    print('LOADING MODEL VERSION ' + str(modelVersion) + '...')
    m_tmp = best_NN.read('connect4', runNumber, best_player_version)
        
    best_NN.model.set_weights(m_tmp.get_weights())
    print('done')
    
    return Agent('best_player', 42,
                      42, config.MCTS_SIMS, config.CPUCT, best_NN)
    
    