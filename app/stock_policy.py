import numpy as np

class StockPolicy():
    
    def __init__(self, epsilon = .03):
        self._action = lambda position: 3 if position == 0 else 2
        self._epsilon = epsilon
        self._keyAction = {
            0 : ['B', 'K', 'S'],
            1 : ['C', 'K']
        }

    def policyFunction(self, state, Q, episode = 1):
        
        position = state[-1]
        
        epsilon = self._epsilon / episode

        probabilities = np.ones(self._action(position), dtype = float) * (epsilon / self._action(position))

        best = np.argmax(Q[state][0:self._action(position)])
        
        probabilities[best] += (1.0 - epsilon)

        return probabilities
    
    def bestAction(self, state, Q):
        position = state[-1]
        best = np.argmax(Q[state][0:self._action(position)])
        return best


    def nameAction(self, position, action):
        return self._keyAction[position][action]