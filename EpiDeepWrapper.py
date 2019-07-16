# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 15:53:14 2018

@author: Bijaya
"""

import torch 
from torch.autograd import Variable
from clustering_datasets import load_mydata
from clustering_datasets import load_RNNdata
import numpy as np
from deepClusteringwCL import DeepClustering
from deepClusteringwCLTime import DeepClusteringTime

def epideep(args):

    #read the input parameters
    start_week  = args.start_week   #To make better predictions, the start week should be > 20
    end_week    = args.end_week     #For future incidence predictions, the end_week should be < 52-N
    start_year  = args.start_year
    end_year    = args.end_year
    pred_metrics= args.pred_metrics       #four eval_metrics: future-inci, peak, peak-time, onset time
    eval_metrics= args.eval_metrics       #three evaluation metrics: RMSE, MAPE, log score.
    region      = args.region             #input regions
    if region == 'National':
        region = 'X'
    iterations  = args.iterations

    list_of_lens = [i for i in range(start_week,end_week)]
    list_of_rmse = []
    list_of_mape = []
    new_data = []
    new_prediction = []
    new_count = -1
    peak_prediction=[]
    peak_prediction_logscore=[]
    peak_time_prediction=[]
    onset_time_prediction=[]
    if pred_metrics == "Future":
        method = 0
    elif pred_metrics == "Peak":
        method = 1
    elif pred_metrics == "Peak_Time":
        method = 2
    elif pred_metrics == "Onset":
        method = 3
    else:
        print("Unavailable prediction metrics! Please use Future, Peak, Peak_Time, or Onset!")
        quit()

    for length in list_of_lens:
        prediction_num = 52 - length

        first_year = start_year

        current_year = end_year-1   #the prediction is actually on year not year-1

        training_data_num = current_year - first_year+1 #diff1

        #########################################################################
        # Get clustering data
        #########################################################################
        full_length_data  = load_mydata(52, first_year, region)

        train_full_length_data = full_length_data[:training_data_num]

        #########################################################################
        # Get clustering data
        #########################################################################
        query_length_data  = load_mydata(length, first_year, region)
        
        train_query_length_data = query_length_data[:training_data_num]

        test_query_length_data= query_length_data[training_data_num:training_data_num+1]

        rnn_data, rnn_label_wILI, rnn_label_peak, rnn_label_peak_time, rnn_label_onset_time = load_RNNdata(length, first_year,region)

        if method == 0:
            rnn_label = rnn_label_wILI
        if method == 1:
            rnn_label = rnn_label_peak
        if method == 2:
            rnn_label = rnn_label_peak_time
        if method == 3:
            rnn_label = rnn_label_onset_time
        
        train_rnn_data = rnn_data[:training_data_num]
        train_rnn_label = rnn_label[:training_data_num]

        test_rnn_data = rnn_data[training_data_num:training_data_num+1]

        #########################################################################
        # CLuster
        #########################################################################

        #diff2, updated
        if method <= 1:
            clustering = DeepClustering(query_length_data.shape[1],  20, full_length_data.shape[1], 20, 4)
            clustering.fit(train_query_length_data, train_full_length_data, train_rnn_data, train_rnn_label, num_epoch = iterations)

        if method>= 2:
            clustering = DeepClusteringTime(query_length_data.shape[1],  20, full_length_data.shape[1], 20, 4, output_size = rnn_label.shape[1] )
            clustering.fit(train_query_length_data, train_full_length_data, train_rnn_data, train_rnn_label, num_epoch = iterations)


        #########################################################################
        # Predict and calculate RMSE
        #########################################################################
        if method == 0:
            epoch=0
            sumval=0    #for RMSE
            sumval2=0   #for MAPE
            nextN=4
            while epoch <= nextN-1:# 0 -- nextN-1 is nextN predictions
                if epoch == 0:
                    new_data.append([])
                    new_prediction.append([])
                    new_count+=1
                pred = clustering.predict(test_query_length_data, test_rnn_data)
                rest_query = test_query_length_data[0][1:length]
                
                #get new
                #query_length_data  = load_mydata(length+epoch, first_year, region)
                new_query_point = (query_length_data[training_data_num:training_data_num+1])[0][-1]

                #add new
                new_query = torch.cat((torch.tensor(rest_query), torch.DoubleTensor([new_query_point]) ),0)
                test_query_length_data[0] = new_query

                #remove first element in test_rnn_data and add next value from the pred
                rest_rnn_query = test_rnn_data[0][1:length]
                
                #get new
                new_rnn_query_point = pred
                #add new
                new_rnn_query = torch.cat((torch.tensor(rest_rnn_query), torch.DoubleTensor([[new_rnn_query_point]]) ),0)
                test_rnn_data[0] = new_rnn_query
                
                sumval += (float(new_rnn_query_point)-float(new_query_point))**2
                sumval2+= abs((float(new_rnn_query_point)-float(new_query_point))/float(new_query_point))
                new_data[new_count].append(float(new_query_point))
                new_prediction[new_count].append(float(new_rnn_query_point))
                epoch+=1

            sumval = sumval/(epoch) #RMSE =sqrt(sum(val^2)/N)
            sumval2 = sumval2/(epoch) #MAPE = sum((pred-data)/data)/N
            list_of_rmse.append(sumval)
            list_of_mape.append(sumval2)
        if method == 1:
            pred = clustering.predict(test_query_length_data, test_rnn_data)
            peak_prediction.append(float(pred))
            pred = (Variable(pred).data).cpu().numpy()
            peak_prediction_logscore.append(pred[0])
        if method == 2:
            pred = clustering.predict(test_query_length_data, test_rnn_data)
            pred = (Variable(pred).data).cpu().numpy()
            peak_time_prediction.append(pred[0])
        if method == 3:
            pred = clustering.predict(test_query_length_data, test_rnn_data)
            pred = (Variable(pred).data).cpu().numpy()
            onset_time_prediction.append(pred[0])

    #########################################################################
    # Outputs the results
    #########################################################################
    if eval_metrics == 'logscore':
        if method == 0:
            print("Now predicting the wILI value for year ",current_year+1," and week ",length)
            f = open("./results/Future_"+str(current_year+1)+".txt", "w")
            Outputs=new_prediction
        if method == 1:
            print("Now predicting the peak value for ",current_year+1," and week ",length)
            f = open("./results/Peak_"+str(current_year+1)+".txt", "w")
            Outputs=peak_prediction_logscore
        if method == 2:
            print("Now predicting the peak time for ",current_year+1," and week ",length)
            f = open("./results/Peak_Time_"+str(current_year+1)+".txt", "w")
            Outputs=peak_time_prediction
        if method == 3:
            print("Now predicting the onset time for ",current_year+1," and week ",length)
            f = open("./results/Onset_"+str(current_year+1)+".txt", "w")
            Outputs=onset_time_prediction


        if method <=1:  #for peak and future
            for i,j in zip(list_of_lens, Outputs):
                #output the point predictions
                f.write("Point"+","+"Value"+","+str(0.05+0.1*j.tolist().index(max(j)))+"\n" )
                #output the binned predictions
                countOut=0
                for k in j:
                    beginBin    =0.1*countOut
                    endBin      =0.1*countOut+0.1
                    if countOut == len(list(Outputs))-1:    #last bin is [13,100)
                        endBin = 100
                    f.write(str(beginBin)+","+str(endBin)+","+str(k)+"\n")
                    countOut+=1
        if method >1:   #for peak time and onset time
            for i,j in zip(list_of_lens, Outputs):
                #output the point predictions
                f.write("PointValue"+","+str(i)+","+str(20+j.tolist().index(max(j)))+"\n" )
                #output the binned predictions
                countOut=0
                for k in j:
                    if countOut+40 <=52:
                        beginBin    =countOut+20
                        endBin      =countOut+21
                    elif method == 3 and countOut == len(list(Outputs))-1:
                        #the last bin for onset is the no-onset bin.
                        beginBin = 'none'
                        endBin  ='none'
                    elif countOut+40 >52:     #need a mod function
                        beginBin    =(countOut+20)%52
                        endBin      =(countOut+21)%52
                
                    f.write(str(beginBin)+","+str(endBin)+","+str(k)+"\n")
                    countOut+=1
            f.close()
    elif eval_metrics == 'RMSE' or eval_metrics == 'MAPE':
        if method == 0:
            RMSE_individual     = np.sqrt(list_of_rmse)
            RMSE_total          = np.sqrt(sum(list_of_rmse)*nextN/(nextN*len(list_of_rmse)))
            MAPE_individual     = list_of_mape
            MAPE_total          = sum(list_of_mape)*nextN/(nextN*len(list_of_mape))
            if eval_metrics == "RMSE":
                final_metrics_indi  = RMSE_individual
                final_metrics_total = RMSE_total
            elif eval_metrics == "MAPE":
                final_metrics_indi  = MAPE_individual
                final_metrics_total = MAPE_total

            print("Now predicting the wILI value for year ",current_year+1," and week ",length)
            print("New data = " ,new_data)
            print("New prediction = ", new_prediction)
            print(eval_metrics," for this week = ", final_metrics_indi)
            
            f = open("./results/Future_"+str(current_year+1)+".txt", "w")
            for i,j,k,l in zip(list_of_lens,new_data,new_prediction,final_metrics_indi):
                f.write(str(i)+","+str(j[0])+","+str(j[1])+","+str(j[2])+","+str(j[3])+","+str(k[0])+","+str(k[1])+","+str(k[2])+","+str(k[3])+","+str(l)+'\n')
            f.write("The total "+eval_metrics+" for these weeks is "+str(final_metrics_total)+'\n')
            f.close()
        if method == 1:
            RMSE_total =0
            MAPE_total =0
            data_here = float(rnn_label_peak[training_data_num])
            for i in peak_prediction:
                RMSE_total+=(float(i)-data_here)**2.0
                MAPE_total+=abs(float(i)-data_here)/data_here
            RMSE_total=np.sqrt(RMSE_total/len(list_of_lens))
            MAPE_total=MAPE_total/len(list_of_lens)
            if eval_metrics == "RMSE":
                final_metrics_total = RMSE_total
            elif eval_metrics == "MAPE":
                final_metrics_total = MAPE_total

            print("Now predicting the peak value for ",current_year+1," and week ",length)
            f = open("./results/Peak_"+str(current_year+1)+".txt", "w")
            for i,j in zip(list_of_lens,peak_prediction):
                f.write(str(i)+","+str(data_here)+","+str(j)+'\n')
            f.write("The total "+eval_metrics+" for these weeks is "+str(final_metrics_total)+'\n')
            f.close()
        if method == 2:
            RMSE_total =0
            MAPE_total =0
            #the data is a list with probabilities for each week, peak_time is the week with maximum of them. And 0th is for 20st week so +20.
            data_here = float(20+(rnn_label_peak_time[training_data_num]).tolist().index(max(rnn_label_peak_time[training_data_num])))
            for i in peak_time_prediction:
                prediction_here =20+i.tolist().index(max(i))
                RMSE_total+=(float(prediction_here)-data_here)**2.0
                MAPE_total+=abs(float(prediction_here)-data_here)/data_here
            RMSE_total=np.sqrt(RMSE_total/len(list_of_lens))
            MAPE_total=MAPE_total/len(list_of_lens)
            if eval_metrics == "RMSE":
                final_metrics_total = RMSE_total
            elif eval_metrics == "MAPE":
                final_metrics_total = MAPE_total

            print("Now predicting the peak time for ",current_year+1," and week ",length)
            f = open("./results/Peak_Time_"+str(current_year+1)+".txt", "w")
            for i,j in zip(list_of_lens,peak_time_prediction):
                f.write(str(i)+","+str(data_here)+","+str(20+j.tolist().index(max(j)))+'\n')
            f.write("The total "+eval_metrics+" for these weeks is "+str(final_metrics_total)+'\n')

            f.close()
        if method == 3:
            RMSE_total =0
            MAPE_total =0
            data_here = float(20+(rnn_label_onset_time[training_data_num]).tolist().index(max(rnn_label_onset_time[training_data_num])))
            for i in onset_time_prediction:
                prediction_here =20+i.tolist().index(max(i))
                RMSE_total+=(float(prediction_here)-data_here)**2.0
                MAPE_total+=abs(float(prediction_here)-data_here)/data_here
            RMSE_total=np.sqrt(RMSE_total/len(list_of_lens))
            MAPE_total=MAPE_total/len(list_of_lens)
            if eval_metrics == "RMSE":
                final_metrics_total = RMSE_total
            elif eval_metrics == "MAPE":
                final_metrics_total = MAPE_total

            print("Now predicting the onset time for ",current_year+1," and week ",length)
            f = open("./results/Onset_"+str(current_year+1)+".txt", "w")
            for i,j in zip(list_of_lens,onset_time_prediction):
                f.write(str(i)+","+str(data_here)+","+str(20+j.tolist().index(max(j)))+'\n')
            f.write("The total "+eval_metrics+" for these weeks is "+str(final_metrics_total)+'\n')
            f.close()

    else:
        print("Unavailable evaluation metrics! Please use RMSE, MAPE, or logscore!")
        quit()

    #########################################################################
    # Calculate the embeddings
    #########################################################################
    emd = clustering.embed(query_length_data, rnn_data).data.numpy()
    
    print()

    emd_file = open('results/embedding.txt', 'w')
    
    year_label = start_year
    
    for i in range(len(emd)):
        emd_file.write(str(year_label))
        for vals in emd[i]:
            emd_file.write(",%.10f" % vals)
        emd_file.write('\n')
        year_label+=1
        
        
    emd_file.close()


