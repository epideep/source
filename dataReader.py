import numpy as np

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    
class DataReader():
    
    def __init__( self, window, length):
        self.w = window
        self.l = length
        
    
    def convert_model_to_vec(self, model_id, total_model):
        vec = [0]*total_model
        vec[model_id] = 1
        return vec
    
    def get_data_all(self, trainining_file):
        window = self.w
        length = self.l
        train_file = open(trainining_file,'r')
        all_regions = train_file.readline()
        count = 0
        mapper ={}
        train_X = []
        train_Y_peak = []
        train_Y_peak_time = []
        train_Y_region = []
        train_Next = []
        
        
        for region in all_regions.strip().split(':'):
            mapper[region] = count
            count +=1
        
        for line in train_file:
            raw = line.strip().split(':')
            region = raw[0]
            regionVec = self.convert_model_to_vec(mapper[region], count)
            
            peak_time = float(raw[1])
            peak = float(raw[2])
            
            nextnode = float(raw[3+length+window])
            if(nextnode<=200):
                timeseries = []
                
                for i in range(3, length+3):
                    data = []
                    for j in range(window):
                        data.append(float(raw[i+j]))
                    timeseries.append(data)
                train_X.append(timeseries)
                train_Y_peak.append(peak)
                train_Y_peak_time.append(peak_time)
                train_Y_region.append(regionVec)
                train_Next.append(float(raw[3+length+window]))
            
            
        # train_X, train_Y_peak,  train_Y_peak_time, train_Y_region, 
        return ( train_X, train_Y_peak,  train_Y_peak_time, train_Y_region, train_Next)
    
    
    
    
    def get_data_next_train_emd(self, training_file):
        window = self.w
        length = self.l
        train_file = open(training_file,'r')
        all_regions = train_file.readline()
        count = 0
        mapper ={}
        train_X = []
        train_Y_peak = []
        train_Y_peak_time = []
        train_Y_region = []
        train_Next = []
        emd = []
        
        for region in all_regions.strip().split(':'):
            mapper[region] = count
            count +=1
        
        for line in train_file:
            raw = line.strip().split(':')
            region = raw[0]
            regionVec = self.convert_model_to_vec(mapper[region], count)
            
            peak_time = float(raw[1])
            peak = float(raw[2])
            
            y_emd = []
            for ind in range(3,18+3):
                y_emd.append(float(raw[ind]))
            
            cur = 0
            while(length+cur+window < len(raw)-3):
                nextnode = float(raw[3+length+window+cur])
                if(nextnode<=20000000):
                    timeseries = []
                
                    for i in range(cur+3, cur+length+3):
                        data = []
                        for j in range(window):
                            data.append(float(raw[i+j]))
                            
                        timeseries.append(data)
                train_X.append(timeseries)
                emd.append(y_emd)
                train_Next.append(float(raw[3+length+window+cur]))
                cur +=1
            
            
        # train_X, train_Y_peak,  train_Y_peak_time, train_Y_region, 
        return ( np.array(train_X), np.array(train_Next), emd)
    
    def get_data_next_test(self, training_file):
        window = self.w
        length = self.l
        train_file = open(training_file,'r')
        all_regions = train_file.readline()
        count = 0
        mapper ={}
        train_X = []
        train_Y_peak = []
        train_Y_peak_time = []
        train_Y_region = []
        train_Next = []
        
        
        for region in all_regions.strip().split(':'):
            mapper[region] = count
            count +=1
        
        for line in train_file:
            raw = line.strip().split(':')
            region = raw[0]
            regionVec = self.convert_model_to_vec(mapper[region], count)
            
            peak_time = float(raw[1])
            peak = float(raw[2])
            
            
           
            nextnode = float(raw[3+length+window])
            if(nextnode<=20000000):
                timeseries = []
            
                for i in range(3, length+3):
                    data = []
                    for j in range(window):
                        data.append(float(raw[i+j]))
                        
                    timeseries.append(data)
            train_X.append(timeseries)
            train_Next.append(float(raw[3+length+window]))
                
            
            
        # train_X, train_Y_peak,  train_Y_peak_time, train_Y_region, 
        return ( np.array(train_X), np.array(train_Next))
    
    
    def get_data_peak_time(self, training_file):
        window = self.w
        length = self.l
        train_file = open(training_file,'r')
        all_regions = train_file.readline()
        count = 0
        mapper ={}
        train_X = []
        train_Y_peak = []
        train_Y_peak_time = []
        train_Y_region = []
        train_Next = []
        
        
        for region in all_regions.strip().split(':'):
            mapper[region] = count
            count +=1
        
        for line in train_file:
            raw = line.strip().split(':')
            region = raw[0]
            regionVec = self.convert_model_to_vec(mapper[region], count)
            
            peak_time = float(raw[2])
            peak = float(raw[2])
            
            nextnode = float(raw[3+length+window])
            if(nextnode<=20000000):
                timeseries = []
                
                for i in range(3, length+3):
                    data = []
                    for j in range(window):
                        data.append(float(raw[i+j]))
                    timeseries.append(data)
                train_X.append(timeseries)
                train_Y_peak.append(peak)
                train_Y_peak_time.append(peak_time)
                train_Y_region.append(regionVec)
                train_Next.append(float(raw[3+length+window]))    
            
            
        # train_X, train_Y_peak,  train_Y_peak_time, train_Y_region, 
        return ( np.array(train_X), np.array(train_Y_peak_time))
    
    
    def get_data_region(self, training_file):
        window = self.w
        length = self.l
        train_file = open(training_file,'r')
        all_regions = train_file.readline()
        count = 0
        mapper ={}
        train_X = []
        train_Y_peak = []
        train_Y_peak_time = []
        train_Y_region = []
        train_Next = []
        
        
        for region in all_regions.strip().split(':'):
            mapper[region] = count
            count +=1
        
        for line in train_file:
            raw = line.strip().split(':')
            region = raw[0]
            
            regionVec = self.convert_model_to_vec(mapper[region], count)
            
            peak_time = float(raw[1])
            peak = float(raw[2])
            
            nextnode = float(raw[3+length+window])
            if(nextnode<=2000):
                timeseries = []
                
                for i in range(3, length+3):
                    data = []
                    for j in range(window):
                        data.append(float(raw[i+j]))
                    timeseries.append(data)
                train_X.append(timeseries)
                train_Y_peak.append(peak)
                train_Y_peak_time.append(peak_time)
                train_Y_region.append(regionVec)
                train_Next.append(float(raw[3+length+window]))    
            
            
        # train_X, train_Y_peak,  train_Y_peak_time, train_Y_region, 
        return ( np.array(train_X), np.array(train_Y_region))
    
    
    def get_data_next_regression_test(self, training_file):
        window = self.w
        length = self.l
        train_file = open(training_file,'r')
        all_regions = train_file.readline()
        count = 0
        mapper ={}
        train_X = []
        train_Y_peak = []
        train_Y_peak_time = []
        train_Y_region = []
        train_Next = []
        
    
        for region in all_regions.strip().split(':'):
            mapper[region] = count
            count +=1
    
        for line in train_file:
            raw = line.strip().split(':')
            region = raw[0]
            if(region != ''):
                regionVec = self.convert_model_to_vec( mapper[region], count)
                
                peak_time = float(raw[1])
                peak = float(raw[2])
                
                nextnode = float(raw[3+length])
                if(nextnode<=20000000):
                    timeseries = []
                    
                    for i in range(3, length+3):
                        timeseries.append(float(raw[i]))
                    #print(len(timeseries), "len")
                    train_X.append(timeseries)
                    train_Y_peak.append(peak)
                    train_Y_peak_time.append(peak_time)
                    train_Y_region.append(regionVec)
                    train_Next.append(float(raw[3+length]))
        
  
        return np.array(train_X), np.array(train_Next)
    
    def get_data_next_regression_train(self, training_file):
        window = self.w
        length = self.l
        train_file = open(training_file,'r')
        all_regions = train_file.readline()
        count = 0
        mapper ={}
        train_X = []
        train_Y_peak = []
        train_Y_peak_time = []
        train_Y_region = []
        train_Next = []
        
    
        for region in all_regions.strip().split(':'):
            mapper[region] = count
            count +=1
    
        for line in train_file:
            raw = line.strip().split(':')
            region = raw[0]
            if(region != ''):
                regionVec = self.convert_model_to_vec( mapper[region], count)
                
                peak_time = float(raw[1])
                peak = float(raw[2])
                
                cur = 0
                while(length+cur+window < len(raw)-3):
                    nextnode = float(raw[3+length+window+cur])
                        
                    timeseries = []
                    
                    for i in range(3+cur, length+3+cur):
                        timeseries.append(float(raw[i]))
                    #print(len(timeseries), "len")
                    train_X.append(timeseries)
                    train_Next.append(float(raw[3+length+window+cur]))
                    cur += 1
        
  
        return np.array(train_X), np.array(train_Next)
    
    
    def get_data_peak_time_regression(self, training_file):
        window = self.w
        length = self.l
        train_file = open(training_file,'r')
        all_regions = train_file.readline()
        count = 0
        mapper ={}
        train_X = []
        train_Y_peak = []
        train_Y_peak_time = []
        train_Y_region = []
        train_Next = []
        
    
        for region in all_regions.strip().split(':'):
            mapper[region] = count
            count +=1
    
        for line in train_file:
            raw = line.strip().split(':')
            region = raw[0]
            if(region != ''):
                regionVec = self.convert_model_to_vec( mapper[region], count)
                
                peak_time = float(raw[1])
                peak = float(raw[2])
                
                nextnode = float(raw[3+length])
                if(nextnode<=20000000):
                    timeseries = []
                    
                    for i in range(3, length+3):
                        timeseries.append(float(raw[i]))
                    #print(len(timeseries), "len")
                    train_X.append(timeseries)
                    train_Y_peak.append(peak)
                    train_Y_peak_time.append(peak_time)
                    train_Y_region.append(regionVec)
                    train_Next.append(float(raw[3+length]))
        
  
        return np.array(train_X), np.array(train_Y_peak_time)
