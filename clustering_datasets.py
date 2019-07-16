import numpy as np
import os

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def load_mydata(length, first_year, data_region, path = './data'):
    
    
    if data_region != 'X': # if not national region
        str_arr = data_region.split('n')
        data_region = str_arr[0]+'n '+str_arr[1]
        
        
    input_file =  os.path.join( path, 'ILINet.csv')
    x = []

    
    # indexed by region
    all_data = {}
    in_f = open(input_file)
    in_f.readline()
    in_f.readline()
    
    for line in in_f:
        raw = line.strip().split(',')
        region = raw[1].strip()
        year = int(raw[2].strip())
        week = int(raw[3].strip())
        ## upto 20th week belongs to last years cycle
        if(week <= 20):
            year -= 1
        infection = raw[4].strip()
        inf = 0
        if is_number(infection):
            inf = float(infection)
        if region not in all_data:
            all_data[region]={}
        if year not in all_data[region]:
            all_data[region][year] = []
        all_data[region][year].append(inf)
        
    indexDic = {}
    
    raw = all_data[data_region]
    keylist = list(raw.keys())
    keylist.sort()
    for year in keylist:
        if year>=first_year and len(raw[year]) == 52:
            indexDic[len(x)] = year
            x.append(raw[year][0:length])
    

    return np.array(x)

def load_RNNdata(length, first_year,  data_region, path = './data'):
    
    if data_region != 'X': # if not national region
        str_arr = data_region.split('n')
        data_region = str_arr[0]+'n '+str_arr[1]
        

    input_file =  os.path.join( path, 'ILINet.csv')
    
    x = []
    y = []
    peak = []
    peak_time = []
    onset_time = []

    
    baseline_file = open(os.path.join(path, 'baseline'))
    cdc_baselines = {}
    
    for line in baseline_file:
        arr = line.strip().split()
        #print(arr)
        year = int(arr[0])
        baseline = float(arr[1])
        cdc_baselines[year] = baseline
    
    
    # indexed by region
    all_data = {}
    in_f = open(input_file)
    in_f.readline()
    in_f.readline()
    
    for line in in_f:
        raw = line.strip().split(',')
        region = raw[1].strip()
        year = int(raw[2].strip())
        week = int(raw[3].strip())
        ## upto 20th week belongs to last years cycle
        if(week <= 20):
            year -= 1
        infection = raw[4].strip()
        inf = 0
        if is_number(infection):
            inf = float(infection)
        if region not in all_data:
            all_data[region]={}
        if year not in all_data[region]:
            all_data[region][year] = []
        all_data[region][year].append(inf)
        
    indexDic = {}
    

    raw = all_data[data_region]
    keylist = list(raw.keys())
    keylist.sort()
    for year in keylist:
        if year>=first_year and len(raw[year]) == 52:
            indexDic[len(x)] = year
            x.append(raw[year][0:length])
            y.append(raw[year][length])
            peak.append(max(raw[year]))
            peak_time_val = (raw[year]).index(max(raw[year]))
            peak_time_vec = [0]*52
            peak_time_vec[peak_time_val] = 1
            peak_time.append(peak_time_vec) #careful the peak time is from the 21st week
                                                                #counts from 0, so 37 means 21+37-52=6 week next year
            onset = -1
            baseline_val = cdc_baselines[year]
            for i in range(len(raw[year])-3):
                trueVals = [raw[year][x]>=baseline_val for x in range(i,i+3)]
                if all(trueVals):
                    onset = i
                    break
            onset_vec = [0]*53
            onset_vec[onset]= 1
            onset_time.append(onset_vec) #careful the peak time is from the 21st week
                                     #counts from 0, so 37 means 21+37-52=6 week next year
                                     # -1 means no onset
            
            
    x = np.array(x)
    x = x[:, :,np.newaxis]

    return x, np.array(y),np.array(peak),np.array(peak_time), np.array(onset_time)
    

