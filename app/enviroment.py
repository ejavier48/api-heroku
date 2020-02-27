from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import json

import multiprocessing

class StockEnviroment:

    def __init__(self, data, window = None):
        self._data = data #dataset, information to generate the states
        self._total = len(data) #number of samples in the data
        self._count = 0 #index for current sample 
        if window is None: 
            self._window = self._total # use total sample
        else:
            self._window = window # window to use during the training
    
    #slide window
    def _slide(self):
        self._index = self._count * self._window
        if self._total < (self._index+self._window):
            self._index = 0
            self._count = 0
        self._count += 1

    #start new episode with same data
    def reset(self):
        self._slide()
        self._state = {}
        self._moves = []
        self._prepareState()
        return self._state

    #receive action, return new state, reward and wheter episode is over or not
    def step(self, action):
        reward = self._prepareState(action)
        return self._state, reward, self._isOver()

    #check if episode is over
    def _isOver(self):
        return self._index == (self._count*self._window)

    def _candleGroup(self, bStick, uWick, lWick):
        if(bStick > uWick):
            if(bStick > lWick):
                if(bStick>(uWick+lWick)):
                    return "bS>(uW+lW)"
                elif(uWick==lWick):
                    return "bS>(uW=lW)"
                elif(uWick>lWick):
                    return "bS>uW>lW"
                else:
                    return "bS>lW>uW"
            elif(bStick==lWick):
                return "(bS=lW)>uW"
            elif(lWick>(bStick+uWick)):
                return "lW>(bS+uW)"
            else:
                return "lW>bS>uW"
        elif(bStick == uWick):
            if(bStick > lWick):
                return "(bS=uW)>lW"
            elif(bStick==lWick):
                return "(bS=uW=lW)"
            elif(lWick>(bStick+uWick)):
                return "lW>(bS+uW)"
            else:
                return "lW>(bS=uW)"
        elif(uWick > lWick):
            if(uWick>(bStick+lWick)):
                return "uW>(bS+lW)"
            elif(bStick>lWick):
                return "uW>bS>lW"
            elif(bStick==lWick):
                return "uW>(bS=lW)"
            else:
                return "uW>lW>bS"
        elif(uWick == lWick):
            return "(uW=lW)>bS"
        elif(lWick>(bStick+uWick)):
            return "lW>(bS+uW)"
        else:
            return "lW>uW>bS"
        #if(uWick > uWick):
        #    if(uWick > bStick):
        #        if(uWick > bStick):
        #            return 1#'UW>LW>CB'
        #        elif(uWick < bStick):
        #            return 2#'UW>CB>LW'
        #        else:
        #            return 3#''UW>LW=CB'
        #    elif(uWick < bStick):
        #            return 4#'CB>UW>LW' #
        #    else:
        #        return 5#'UW=CB>LW' #
        #elif(uWick < uWick):
        #    if(uWick > bStick):
        #        if(uWick > bStick):
        #            return 6#'LW>UW>CB' #
        #        elif(uWick < bStick):
        #            return 7#'LW>CB>UW' 
        #        else:
        #            return 8#'LW>UW=CB'
        #    elif(uWick < bStick):
        #        return 9#'CB>LW>UW'
        #    else:
        #        return 10#'LW=CB>UW'
        #elif(uWick > bStick):
        #    return 11#'UW=LW>CB'
        #elif(uWick < bStick):
        #    return 12#'CB>UW=LW'
        #else:
        #    return 13#'UW=CB=LW'

    #return state after take an action
    def _prepareState(self, action = 'K'):
        #check if episode already finish
        if self._isOver():
            return None

        #get current data 
        current = self._data[self._index]
        
        reward = 0
        
        position = self._state['position'] if 'position' in self._state else {'type':0}
        
        if action != 'K': #if not keep
            if action == 'C': #close position
                reward = position['type'] * (current['Open'] - position['price'])
                position = {'type':0}
            elif action == 'B': #open buy position
                position['type'] = 1
                position['price'] = current['Open']
            elif action == 'S': #open sell position
                position['type'] = -1
                position['price'] = current['Open']
        
        #elements to resume state
        phi = 0.0
        lWick = 0.0
        uWick = 0.0
        bStick = 0.0
        tStick = 0.0
        gain = 0

        if position['type'] != 0:
            aux = position['type'] * (current['Close'] - position['price'])
            gain = 1 if 0 < aux else -1 

        tendency = 0 if current['Close'] < current['Open'] else 1
        
        tStick = current['High'] - current['Low']
        
        if tendency == 0: #bear
            uWick = current['High'] - current['Open']
            lWick = current['Close'] - current['Low']
            bStick = current['Open'] - current['Close']    
        else: #bull
            uWick = current['High'] - current['Close']
            lWick = current['Open'] - current['Low']
            bStick = current['Close'] - current['Open']
        
        if tStick == 0:
            phi = 100
        else:
            phi = round( (bStick/tStick) * 100)
        
        group = self._candleGroup(bStick, uWick, lWick)
        
        nextState = {
            'phi': phi,
            'pattern' : group,
            'tendency': tendency,
            'gain': gain,
            'position' : position
        }
        
        self._index += 1
        self._state = nextState
        return reward

class StockPolicy():
    
    def __init__(self, epsilon = .01):
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

class QLearning():

    def __init__(self, data, episodes, epsilon, gamma, alpha, window = None):
        self._env = StockEnviroment(data, window)
        #self._decay = int(len(data) / window) * 100
        self._decay = 2500
        #print(self._decay)
        self._alpha = alpha
        self._gamma = gamma
        self._epsilon = epsilon
        self._episodes = episodes
        self._policy = StockPolicy(self._epsilon)
        self.Q = defaultdict(lambda: np.zeros(3))
        self._name = "".join(["de", str(epsilon), "_g", str(gamma), "_a", str(alpha), "_"])
    
    def _processState(self, rawState):
        return (rawState['phi'], rawState['pattern'], 
                rawState['tendency'], rawState['gain'], rawState['position']['type'])
    
    def run(self):
        stats = {
            'moves' : [],
            'rewards': [],
            'bestQ' : None,
            'bReward' : None,
            'bEpisode' : None,
        }

        bestQ = None
        bestRw = -1e9
        
        eps = 1

        for i in range(self._episodes):

            moves = 0
            rewards = 0

            state = self._processState(self._env.reset())

            if (i+1)%self._decay == 0:
                eps += .25

            while True:
                actions = self._policy.policyFunction(state, self.Q, eps)

                action = np.random.choice(np.arange(len(actions)), p = actions)

                nState, reward, done = self._env.step(self._policy.nameAction(abs(state[-1]), action))

                nState = self._processState(nState)

                if state[-1] != nState[-1]:
                    moves += 1
                rewards += reward

                bAction = self._policy.bestAction(nState, self.Q)
                tdTarget = reward + self._gamma * self.Q[nState][bAction]
                tdDelta = tdTarget - self.Q[state][action]
                self.Q[state][action] += self._alpha * tdDelta

                if done:
                    break
                
                state = nState

            if bestRw < rewards:
                bestRw = rewards
                bestQ = self.Q.copy()

            stats['moves'].append(moves)
            stats['rewards'].append(rewards)

            if((i+1)%5000 == 0):
                fname = ''.join(['rewards_', self._name, str((i+1)/10000), '.data'])
                with open(fname, 'w') as f:
                    f.write(str(stats['rewards']))
                fname = ''.join(['moves_', self._name, str((i+1)/10000), '.data'])
                with open(fname, 'w') as f:
                    f.write(str(stats['moves']))
                stats['moves'] = []
                stats['rewards'] = []


        stats['bestQ'] = dict(bestQ)
        stats['bestRw'] = bestRw
        
        return stats
    
    def getQ(self):
        return self.Q

def multiAgent(name, data, episodes, epsilon, gamma, alpha, window = None):
    agent = QLearning(data, episodes, epsilon, gamma, alpha, window)
    stats = agent.run()
    q = dict(agent.getQ())

    fname = ''.join([name, '_stats.in'])
    with open(fname, 'w') as f:
        init = str(data[0]['Date'])
        end = str(data[-1]['Date'])
        s = ''.join(['Dates:', init, end, '\n'])
        f.write(s)
        s = '\t'.join(['Len', 'epsilon', 'gamma', 'alpha', '\n'])
        f.write(s)
        s = '\t'.join([str(len(data)), str(epsilon), str(gamma), str(alpha), '\n'])
        f.write(s)
        s = '\t'.join(['Stock', 'Gains', '\n'])
        f.write(s)
        stock = str(data[-1]['Close'] - data[0]['Open'])
        gains = str(stats['bestRw'])
        s = '\t'.join([stock, gains, '\n'])
        f.write(s)
        f.write('************************************************\n')
        for moves, reward in zip(stats['moves'], stats['rewards']):
            episode = ''.join([str(moves), ':', str(reward), '\n'])
            f.write(episode)
    
    q = stats['bestQ']
    newQ = {}
    for k in q:
        newQ[str(k)] = list(q[k])

    fname = ''.join([name, '_', str(stats['bestRw']), '_BestQ.json'])
    with open(fname, 'w') as file:
        json.dump(newQ, file)
    
    return
    
if __name__ == '__main__':
    np.random.seed()
    with open('data.json', 'r') as file:
        data = json.load(file)

    threads = []
    sEps = None
    """
    for i in range(12):
        alpha = round(np.random.random_sample() + .1, 3) * .9
        gamma = round(np.random.random_sample() + .1, 3) * .9
        epsilon = round(np.random.random_sample() + .1, 3) * .9
        print(alpha, gamma, epsilon)
        t = multiprocessing.Process(target=multiAgent, args = ('data'+str(i), data, 100000, epsilon, gamma, alpha, sEps))
        threads.append(t)

    for i in range(len(threads)):
        print("Start:", i)
        threads[i].start()
        if (i+1)%3 == 0:
            for j in range(i-1, i+1):
                print("Working....")
                if threads[j].is_alive():
                    threads[j].join()
    
    for i in range(len(threads)):
        if threads[i].is_alive():
            print("Woring....")
            threads[i].join()

    print("finish eps random")

    threads = []
    """
    data = data[2600:]
    epsilon = .999999#round(np.random.random_sample() + .1, 3) * .9
    files = []
    for i in range(3):
        alpha = np.random.choice(np.arange(.5, 1, .025))
        gamma = np.random.choice(np.arange(.5, 1, .025))
        files.append("".join(["{}_de", str(epsilon), "_g", str(gamma), "_a", str(alpha), "_{}"]))
        multiAgent('data'+str(i), data, 75000, epsilon, gamma, alpha, sEps)
        t = multiprocessing.Process(target=multiAgent, args = ('data'+str(i), data, 75000, epsilon, gamma, alpha, sEps))
        threads.append(t)


    for i in range(len(threads)):
        print("Start:", i)
        threads[i].start()
        if (i+1)%3 == 0:
            for j in range(i-1, i+1):
                print("Working....")
                if threads[j].is_alive():
                    threads[j].join()
    
    for i in range(len(threads)):
        if threads[i].is_alive():
            print("Woring....")
            threads[i].join()

    for file in files:
        fname = file.format('rewards', '{}.data')
        iname = file.format('rewards', '.jpg')
        data = []
        for i in np.arange(0.5, 8, .5):
            print(fname.format(str(i)))
            with open(fname.format(str(i)), 'r') as f:
                tdata = f.read()
                tdata = tdata.replace('[', '')
                tdata = tdata.replace(']', '')
                data += list(map(float, tdata.split(', ')))
        plt.plot(data)
        plt.ylabel('reward')
        plt.xlabel('episode')
        plt.savefig(iname)
        plt.close()

        fname = file.format('moves', '{}.data')
        iname = file.format('moves', '.jpg')
        data = []
        for i in np.arange(0.5, 10.5, .5):
            print(fname.format(str(i)))
            with open(fname.format(str(i)), 'r') as f:
                tdata = f.read()
                tdata = tdata.replace('[', '')
                tdata = tdata.replace(']', '')
                data += list(map(float, tdata.split(', ')))
        plt.plot(data)
        plt.ylabel('moves')
        plt.xlabel('episode')
        plt.savefig(iname)
        plt.close()

    print('End program')