#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 14:31:51 2018

@author: linamezghani

TEST AI POLICY AGAINST OTHER POLICY

"""

import connect_four as cf

def playGame(redPlayer, yellowPlayer, nb_games = 1, display = False) :
    AI_wins = 0
    AI_lost = 0
    AI_draws = 0
    avg_nb_moves = 0
    
    for games in range(nb_games) :
        
        g = cf.newGame(redPlayer,yellowPlayer)
        turn = g.getFirstColor() #The player that start chosen randomly
        print("first player : "+turn)
        
        nb_of_moves = 0
        
        #the game is being played
        while (g.getWinner() == cf.NONE and not g.boardIsFull()) :
            action = g.getMove(turn)
            g.insert(action, turn, display)
            turn = g.getOpponent(turn)
        
        #And the winner is...
        winner = g.getWinner()
        if winner == cf.NONE :
            print("Tie !")
            AI_draws += 1
        elif winner == cf.RED :
            AI_wins += 1
            avg_nb_moves = ((AI_wins - 1)*avg_nb_moves + nb_of_moves)/AI_wins
        else :
            print(winner + ' WON THE GAME')
            AI_lost += 1
    
    print("Total number of games : ",nb_games)
    print("Number of games won by AI : ",AI_wins)
    print("Number of draws : ",AI_draws)
    print("Number of games won by Random : ",AI_lost)



"""________ TEST PARAMETERS _________"""

redPlayer = cf.player('AI',1,29)
yellowPlayer = cf.player('Human')

#Player : - 'AI' for the NN (add parameters runNumber and modelVersion 1,29 for best agent)
#         - 'Human' to play manually
#         - 'Random' for a random policy
#         - 'Minimax' for minimax strategy

nb_games = 1 #Number of games to play

display = True #to display the boards. Put false for a big number of games

playGame(redPlayer,yellowPlayer,nb_games,display)
