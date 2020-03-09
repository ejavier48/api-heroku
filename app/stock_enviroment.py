
class StockEnviroment():
    
    def __init__(self, data):
        self.__data = data #dataset, information to generate the states
        self.__total = len(data) #number of samples in the dataset
        self.__index = 0 #index for current sample
        self.__capital = 1000 #start capital for each episode

    #start new episode with same data
    def reset(self):
        self.__index = 0
        self.__state = {}
        self.__moves = []
        self.__money = self.__capital
        self.__prepare_state()
        return self.__state

    #receive action, return new state, reward and wheter episode is over or not
    def step(self, action):
        reward = self.__prepare_state(action)
        if self.__lost_capital():
            return self.__state, 0, True
        return self.__state, reward, self.__is_over()
    
    def __lost_capital(self):
        return (self.__money < 5) and (self.__state['position'] == 0)

    #check if episode is over
    def __is_over(self):
        return self.__index == self.__total or self.__lost_capital()

    def __candle_group(self, bStick, uWick, lWick):
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
    def __prepare_state(self, action = 'K'):
        
        #check if episode already finish
        if self.__is_over():
            return None

        #get current stick 
        current = self.__data[self.__index]
        
        position = self.__state['position'] if 'position' in self.__state else {'type':0}
        
        if action != 'K': #if action different to Keep

            if action == 'C': #close position
                #get the price difference and multiply it for the number of stock bought
                self.__money += position['type'] * (position['num'] * (current['Open'] - position['price'])) 
                position = {'type':0}
                
            elif action == 'B': #open buy position
                position['type'] = 1 #Label as BUY position
                position['price'] = current['Open'] #price when the stock was bought
                position['num'] = int(self.__money/position['price']) #num of stock bought

                if(position['num'] <= 0): #If not enought money, cancel position
                    position = {'type':0}
                else:
                    self.__money -= position['num']*position['price'] #money available

            elif action == 'S': #open sell position
                position['type'] = -1 #Label as SELL position
                position['price'] = current['Open'] #price when the stock was bought
                position['num'] = int(self.__money/position['price']) #num of stock bought

                if(position['num'] <= 0): #If not enought money, cancel position
                    position = {'type':0}
                else:
                    self.__money -= (position['num']*position['price']) #money available
        
        #elements to resume state
        
        #Element get from Open, Close, High, Low prices
        lWick = 0.0 #Lower wick
        uWick = 0.0 #Upper Wick
        bStick = 0.0 #Stick body
        tStick = 0.0 #Stick Length
        phi = 0.0 #Ratio between Stick Length and Body
        
        gain = 0 #Indicate wheter is gaining or lossing

        #If there is a position open, check gain state
        if position['type'] != 0:
            aux = position['type'] * (current['Close'] - position['price'])
            gain = 1 if 0 < aux else -1 
        
        #Check if the candle moved up or down
        tendency = 0 if current['Close'] < current['Open'] else 1
        
        #Get the stick's length
        tStick = current['High'] - current['Low']
        
        if tendency == 0: #bearish or down stick
            #Upper Wick = Higher price - Open price
            uWick = current['High'] - current['Open']
            #Lower Wick = Close price - Lower price
            lWick = current['Close'] - current['Low']
            #Stick body = Open price - Close price
            bStick = current['Open'] - current['Close']    
        else: #bullish or up stick
            #Upper Wick = Higher price - Close price
            uWick = current['High'] - current['Close']
            #Lower Wick = Open price - Lower price
            lWick = current['Open'] - current['Low']
            #Stick body = Close price - Open price
            bStick = current['Close'] - current['Open']
        
        #Calculate phi
        if tStick == 0: #This happens when there are no moves in the market
            phi = 10
        else: #Get the proportion of the Stick Body to its length
            phi = round( (bStick/tStick) * 10)
        
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
        return self.__money#-self.__capital)
