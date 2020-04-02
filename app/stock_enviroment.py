
class StockEnviroment():
    
    def __init__(self, data, capital = 1000.0):
        self.__data = data #dataset, information to generate the states
        self.__total = len(data) #number of samples in the dataset
        self.__index = 0 #index for current sample
        self.__capital = capital #start capital for each episode

    #start new episode with same data
    def reset(self) -> dict:
        self.__index = 0
        self.__state = {}
        self.__moves = []
        self.__money = self.__capital
        self.__prepare_state()
        return self.__state

    #receive action, return new state, reward and wheter episode is over or not
    def step(self, action) -> (dict, float, float, bool):
        current_reward = self.__prepare_state(action)
        if self.__lost_capital():
            return self.__state, current_reward, self.__money, True
        return self.__state, current_reward, self.__money, self.__is_over()
    
    def __lost_capital(self) -> bool:
        return (self.__money < 5) and (self.__state['position']['type'] == 0)

    #check if episode is over
    def __is_over(self) -> bool:
        return self.__index == self.__total or self.__lost_capital()

    def __candle_group(self, bStick, uWick, lWick) -> int:
        if(bStick > uWick):
            if(bStick > lWick):
                if(bStick>(uWick+lWick)):
                    return 1#"bS>(uW+lW)"
                elif(uWick==lWick):
                    return 2#"bS>(uW=lW)"
                elif(uWick>lWick):
                    return 3#"bS>uW>lW"
                else:
                    return 4#"bS>lW>uW"
            elif(bStick==lWick):
                return 5#"(bS=lW)>uW"
            elif(lWick>(bStick+uWick)):
                return 6#"lW>(bS+uW)"
            else:
                return 7#"lW>bS>uW"
        elif(bStick == uWick):
            if(bStick > lWick):
                return 8#"(bS=uW)>lW"
            elif(bStick==lWick):
                return 9#"(bS=uW=lW)"
            elif(lWick>(bStick+uWick)):
                return 6#"lW>(bS+uW)"
            else:
                return 10#"lW>(bS=uW)"
        elif(uWick > lWick):
            if(uWick>(bStick+lWick)):
                return 11#"uW>(bS+lW)"
            elif(bStick>lWick):
                return 12#"uW>bS>lW"
            elif(bStick==lWick):
                return 13#"uW>(bS=lW)"
            else:
                return 14#"uW>lW>bS"
        elif(uWick == lWick):
            return 15#"(uW=lW)>bS"
        elif(lWick>(bStick+uWick)):
            return 6#"lW>(bS+uW)"
        else:
            return 16#"lW>uW>bS"

    #return state after take an action
    def __prepare_state(self, action = 'K') -> float:
        current_reward = 0.0
        
        #check if episode already finish
        if self.__is_over():
            return None

        #get current stick 
        current = self.__data[self.__index]
        
        position = self.__state['position'] if 'position' in self.__state else {'type':0}
        
        if action != 'K': #if action different to Keep

            if action == 'C': #close position
                price_diff = position['type'] * (current['open'] - position['price'])
                
                gainigs = position['price'] + price_diff
                
                #get the price difference and multiply it for the number of stock bought
                
                self.__money += position['num'] * gainigs
                
                current_reward = price_diff
                
                if(0 < current_reward):
                    current_reward = current_reward  * (1.5) #boost for current reward, imitating traders when get gainigs
                
                position = {'type':0}
                
            elif action == 'B': #open buy position
                position['type'] = 1 #Label as BUY position
                position['price'] = current['open'] #price when the stock was bought
                position['num'] = int(self.__money/position['price']) #num of stock bought

                if(position['num'] < 1): #If not enought money, cancel position
                    position = {'type':0}
                else:
                    self.__money -= position['num'] * position['price'] #money available

            elif action == 'S': #open sell position
                position['type'] = -1 #Label as SELL position
                position['price'] = current['open'] #price when the stock was bought
                position['num'] = int(self.__money/position['price']) #num of stock bought

                if(position['num'] < 1): #If not enought money, cancel position
                    position = {'type':0}
                else:
                    self.__money -= (position['num']*position['price']) #money available
                    
        #elements to resume state
        
        #Element get from open, close, high, low prices
        lWick = 0.0 #lower wick
        uWick = 0.0 #Upper Wick
        bStick = 0.0 #Stick body
        tStick = 0.0 #Stick Length
        phi = 0.0 #Ratio between Stick Length and Body
        
        gain = 0 #Indicate wheter is gaining or lossing

        #If there is a position open, check gain state
        if position['type'] != 0:
            #get value for current invesment 
            price_diff = position['type'] * (current['open'] - position['price'])
            gain = 1 if 0 < price_diff else -1 
            
        #Check if the candle moved up or down
        tendency = 0 if current['close'] < current['open'] else 1
        
        #Get the stick's length
        tStick = current['high'] - current['low']
        
        if tendency == 0: #bearish or down stick
            #Upper Wick = higher price - open price
            uWick = current['high'] - current['open']
            #lower Wick = close price - lower price
            lWick = current['close'] - current['low']
            #Stick body = open price - close price
            bStick = current['open'] - current['close']    
        else: #bullish or up stick
            #Upper Wick = higher price - close price
            uWick = current['high'] - current['close']
            #lower Wick = open price - lower price
            lWick = current['open'] - current['low']
            #Stick body = close price - open price
            bStick = current['close'] - current['open']
        
        #Calculate phi
        if tStick == 0: #This happens when there are no moves in the market
            phi = 100
        else: #Get the proportion of the Stick Body to its length
            phi = round( (bStick/tStick) * 100)
        
        #Get the stick's group (1-16)       
        group = self.__candle_group(bStick, uWick, lWick)
        
        #Generate state with the information got above
        nextState = {
            'phi': phi,
            'pattern' : group,
            'tendency': tendency,
            'gain': gain,
            'position' : position
        }

        #move to next stick
        self.__index += 1
        #set new state
        self.__state = nextState
        #return reward
        return current_reward

    def set_state(self, state):
        self.__state = state