import json
from enviroment import StockEnviroment
from enviroment import StockPolicy
from collections import defaultdict
from numpy import array


class ProcessPolicy():

    def __init__(self):
        self._change = lambda x: (int(x[0]), x[1].replace("\'", ""), int(x[2]), int(x[3]), int(x[4]))
    
    def getPolicy(self, fname):
        Q = defaultdict(lambda: array([1, 0, 0]))
        with open(fname, 'r') as file:
            tempQ = json.load(file)
            for k in tempQ.keys():
                key = k[1:-1].split(', ')
                Q[self._change(key)] = tempQ[k]
        return Q

class Agent():

    def __init__(self, data, Q):
        self._env = StockEnviroment(data)
        self._policy = StockPolicy()
        self.Q = Q
    
    def _processState(self, rawState):
        return (rawState['phi'], rawState['pattern'], 
                rawState['tendency'], rawState['gain'], rawState['position']['type'])
    
    def run(self):

        eps = 1


        moves = []
        tmove = {
            'type': None,
            'start': None,
            'end': None, 
            'gain': None
        }

        state = self._processState(self._env.reset())

        index = 0

        while True:

            actions = self._policy.policyFunction(state, self.Q, eps)

            action = self._policy.bestAction(state, self.Q)

            #action = np.random.choice(np.arange(len(actions)), p = actions)
            kAction = self._policy.nameAction(abs(state[-1]), action)
            
            nState, reward, done = self._env.step(kAction)

            if kAction != 'K':
                if kAction == 'C':
                    if reward > 0:
                        tmove['end'] = index
                        tmove['gain'] = reward
                        moves.append(tmove)
                        tmove = {}
                else:
                    tmove['type'] = kAction
                    tmove['start'] = index

            nState = self._processState(nState)
            if done:
                break
            state = nState
            index += 1
            
        return {'moves': moves}
    
    def getQ(self):
        return self.Q


if __name__ == "__main__":
    
    Q = {} 
    qfile = 'data10_251.922485_BestQ.json'
    rPolicy = ProcessPolicy()
    Q = rPolicy.getPolicy(qfile)
    with open('data.json', 'r') as file:
        data = json.load(file)

    agent = Agent(data[0:200], Q)
    print(agent.run())