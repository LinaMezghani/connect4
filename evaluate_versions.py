#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 19:12:54 2018

@author: linamezghani

RUN THIS FILE TO PLOT THE PERCENTAGE OF WIN PER VERSION

"""
import matplotlib.pyplot as plt
import connect_four as cf



"""________ TEST PARAMETERS _________"""

nb_games = 100 #Number of games to play


def test_version(versionNumber) :
    redPlayer = cf.player('AI',1,versionNumber)
    yellowPlayer = cf.player('Minimax')
    
    #Player : - 'AI' for the NN (add parameters runNumber and modelVersion 3,29 for best_NN)
    #         - 'Human' to play manually
    #         - 'Random' for a random policy
    #         - 'Minimax' for minimax strategy
    
    
    
    
    AI_wins = 0
    AI_lost = 0
    AI_draws = 0
    
    for games in range(nb_games) :
        
        g = cf.newGame(redPlayer,yellowPlayer)
        turn = g.getFirstColor() #The player that start chosen randomly
        
        
        #the game is being played
        while (g.getWinner() == cf.NONE and not g.boardIsFull()) :
            action = g.getMove(turn)
            g.insert(action, turn, False)
            turn = g.getOpponent(turn)
        
        #And the winner is...
        winner = g.getWinner()
        if winner == cf.NONE :
            print("Tie !")
            AI_draws += 1
            #g.printBoard()
        elif winner == cf.RED :
            AI_wins += 1
        else :
            AI_lost += 1
            #g.printBoard()
            
    print("version number : ",versionNumber, " number of wins : ", AI_wins)
    return AI_wins

AI_wins = []
for versionNumber in range(1,30) :
    AI_wins.append(test_version(versionNumber))

versions = range(1,30)

print(AI_wins)

plt.plot(versions,AI_wins)
