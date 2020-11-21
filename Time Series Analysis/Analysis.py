# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 15:04:55 2020

@author: Ethan Bosworth

A piece of work for university originally written in MATLAB but converted to Python

"""

#%% import modules

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

pd.options.mode.chained_assignment = None
#%% import data
#import the data using only the closing data

bcs = pd.read_csv("Raw_data/BCS.csv",usecols = [0,4])
lyg = pd.read_csv("Raw_data/LYG.csv",usecols = [0,4])
hsbc = pd.read_csv("Raw_data/HSBC.csv",usecols = [0,4])
rbs = pd.read_csv("Raw_data/RBS.csv",usecols = [0,4])

#%% data exploration
def date(data): #create function to convert the date to datetime and set to the index
    data["Date"] = pd.to_datetime(data["Date"]).dt.date
    data = data.set_index("Date")
    return data

bcs_date = date(bcs) # use the first bank to create a time list
bank_list_rem = [lyg,hsbc,rbs] #save the other banks for later
sdate = bcs_date.index[0] #create a start date
edate = bcs_date.index[-1] #create an end date
data = pd.DataFrame(pd.date_range(sdate,edate,freq='d')) #create a date range between the start and end each day
data.columns = ["Date_time"]
data["Date"] = data["Date_time"].dt.date #create a date containing only the date
data["day"] = data["Date_time"].dt.dayofweek #create a column with the day number
data = data.drop("Date_time",axis = 1)
data = data.set_index("Date")
data = pd.concat([data,bcs_date],axis = 1) 

for i in bank_list_rem: #run the function on the remaining banks and bring all together
    data = pd.concat([data,date(i)],axis = 1)
data.columns = ["day","bcs","lyg","hsbc","rbs"]

data = data[~data["day"].isin([5,6])] #return only dates that are not weekends

null_data = data[data.isnull().any(axis=1)] #inspect any missing data
print(null_data.head())
#looking at the dataframe all of the banks missing days are on the same day
#this indicates that all of these missing data would be some kind of holiday
print(null_data["day"].value_counts()) #as can be seen most are on monday which signals a bank holiday
print(len(null_data)) # 36 days means 9 non-weekend days closed per year which is roughly correct for banks 

#%% filling the missing data

data = data.drop("day",axis = 1)
#forward fill the missing data using the last avaliable data
data = data.fillna(method = "ffill")

#%% plotting the data 

#%%% raw data
banks = ["BCS","LYG","HSBC","RBS"] #create a list of bank names

for i in data:
    a = sns.lineplot(data = data,x = data.index ,y = i) #plot each bank
a.legend(labels = banks)
a.set_title("Closing Price of Banks")
a.set_ylabel("Price")
plt.show()


#%%% Transformed data

log_returns = data.copy()

for i in range(1,len(log_returns)):
    diff = data.iloc[i]/data.iloc[(i-1)]
    log_returns.iloc[i] = np.log(diff)
log_returns = log_returns.drop(log_returns.index[0])

var = np.var(log_returns,axis = 0)
mean = np.mean(log_returns,axis = 0)

log_returns_normal = (log_returns - mean)/np.sqrt(var)

pal = sns.color_palette()
pal = pal.as_hex()
j = len(log_returns_normal.columns)
f, axes = plt.subplots(j,figsize=(12,12))
for i in range(j):
    a = sns.lineplot(data = log_returns_normal,x = log_returns_normal.index ,y = log_returns_normal.columns[i], ax=axes[i],color = pal[i]) #plot each bank
    a.set_ylim(-15,5)
    a.set_title("Closing Price log of " + str(log_returns_normal.columns[i]))
    a.set_ylabel("Price_log")
plt.show()

del a
#%% segmentation
#a short function to preprocess the data
def preprocess(log_data,bank):
    #create a new dataframe from just the specific bank needed
    data = pd.DataFrame(log_data[bank]).reset_index()
    #change the index to be integers and not dates
    data.drop("index",axis = 1,inplace = True)
    #create an array of segments by creating a list of pairs
    range_data = np.array(range(len(data)))
    #add both segment and estimated data to the processed data
    data["segment"] = (range_data/2).astype(int)
    data["estimated"] = data[bank] # estimated at first will be the same as the bank data
    return data

def bottom_up(data,bank,mse): # create a function for the bottom-up segmentation
    #function takes the log_transformed data, the bank to use and the mse to stop at
    #----setup----
    data_output = preprocess(data,bank) # first preprocess the data
    
    unique = np.unique(data_output["segment"]) # create a list of unique segments
    index_list = [] # create an empty list of indexes
    for i in unique: 
        #for each segment find the corresponding index in the main data and add it to the list
        index_list.append([data_output[data_output["segment"] == i].index])
    #convert the list to a dataframe and set the column title
    index_list = pd.DataFrame(index_list)
    index_list.columns = ["index_of_segment"]
    index_list["error"] = 0.0 # create an error column set to 0 for now 
    
    #----first iteration----#
    for i in index_list.index.to_list():
    #for loop will find the error of merging segment i with segment i+1
        if i == len(index_list) - 1: # set the error of the final segment to a large number
            index_list["error"][i] = 1000 # as there is nothing to merge with
        else:
            ind = index_list["index_of_segment"][i].append(index_list["index_of_segment"][i+1])
            #find the index of segment i and add the index of segment i+1
            #which is equivalent to merging the segments
            
            #will find the equation of a line between the start and end points in the index
            y1 = data_output["estimated"][ind[0]]
            x1 = ind[0]
            y2 = data_output["estimated"][ind[-1]]
            x2 = ind[-1]
            m = (y2-y1)/(x2-x1)
            c = y1 - (m*x1)
            estimates = (m*ind)+c
            #will create a list of estimates across the index provided
            
            index_list["error"][i] = ((estimates - data_output["estimated"][ind])**2).mean()
            #update the error in the index dataframe to be the mse of merging
            
    error = np.mean((data_output[bank] - data_output["estimated"])**2)
    #create an initial error from the main data, should be 0 but limitations of computer mean it is not exact
    
    #----main iterations---
    while error < mse:
        #whilst error is under threshold given
        
        min_err = index_list["error"].min()
        min_ind = index_list[index_list["error"] == min_err]
        #find the minimum in the list of errors and find the slice where it occurs
        
        index_list["index_of_segment"][(min_ind.index)[0]] = index_list["index_of_segment"][(min_ind.index)[0]].append(index_list["index_of_segment"][(min_ind.index+1)[0]])
        #at first join the indexes of segment i with segment i+1 and set to the index list
        index_list = index_list.drop((min_ind.index+1)[0])
        #drop the now useless segment i+1
        index_list = index_list.reset_index().drop("index",axis = 1)
        #reset the index of index list to avoid problems with the missing values
        
        
        ind = index_list["index_of_segment"][(min_ind.index)[0]]
        #find the indexes in the main data of the segments that were merged
        data_output["segment"][ind] = np.round(data_output["segment"][ind].mean()).astype("int")
        #set the merged segments to havethe same segment number in the main data
        
        y1 = data_output["estimated"][ind[0]]
        x1 = ind[0]
        y2 = data_output["estimated"][ind[-1]]
        x2 = ind[-1]
        m = (y2-y1)/(x2-x1)
        c = y1 - (m*x1)
        estimates = (m*ind)+c
        
        data_output["estimated"][ind] = estimates
        #find the estimated values for the merged segments and set to the main data for that segment
        
        for i in [min_ind.index[0]-1,min_ind.index[0]]:
            #same calculation of error as above but only to update for the now merged 
            #segment and the previous segment that will merge with it to avoid recalculating if not needed
            if i == len(index_list) - 1:
                index_list["error"][i] = 1000
            else: 
        
                ind = index_list["index_of_segment"][i].append(index_list["index_of_segment"][i+1])
                y1 = data_output["estimated"][ind[0]]
                x1 = ind[0]
                y2 = data_output["estimated"][ind[-1]]
                x2 = ind[-1]
                m = (y2-y1)/(x2-x1)
                c = y1 - (m*x1)
                estimates = (m*ind)+c
                index_list["error"][i] = ((estimates - data_output["estimated"][ind])**2).mean()
        
        #update the mse of the estimated data which is just the original data with any merged segments
        error = np.mean((data_output[bank] - data_output["estimated"])**2)
    return data_output
    

#%% iteration of segmentaiton

#create a list of thresholds to use
thresholds = [0.04,0.1,0.25]
colour = 0 # create a colour variable to keep consistent colours for each bank

for k in banks: # iterate over the list of banks

    k = k.lower() # set to lowercase
    j = len(thresholds) # find the size of the thresholds
    f, axes = plt.subplots(j,figsize=(12,12)) # create j subplots based on the size
    
    for i in range(j): # iterate over the thresholds
        bcs_segment = bottom_up(log_returns_normal,k,thresholds[i])
        #find the bottom_up segmentation for the bank and the threshold
        #and plot the data
        a = sns.lineplot(data = bcs_segment,x = bcs_segment.index ,y = "estimated", ax=axes[i],color = pal[colour]) #plot each bank
        a.set_ylim(-15,5)
        a.set_title("Closing Price log of " + k + " with error " + str(thresholds[i]))
        a.set_ylabel("Price_log")
    plt.show()
    colour = colour + 1

#%%  Prediction linear regression
#creating a linear predictor for a target bank based on the others

#first create a target dataframe consisting of the target bank
y = pd.DataFrame(log_returns_normal["bcs"])
y = y.reset_index().drop("index",axis = 1)
target = y.copy() # create a copy to be used in the predictor
y = y.drop(0) # drop first as we are predicting only day+1 using day
y = y.reset_index().drop("index",axis = 1)


#create a supporting dataframe with the remaining banks and original
supporting = log_returns_normal.drop("bcs",axis = 1)
supporting = supporting.reset_index().drop("index",axis = 1)
X = pd.concat([target,supporting],axis = 1)
X = X.drop(len(X)-1) # drop final as we are predicting day+1 using day

prediction = pd.DataFrame(X["bcs"]) # create a prediction variable to store predictions

#create a funcion to predict values and find the mse
def predict(y,X,a_values):
        #find a prediction based on the followinr formula
        # g_pred(x+1) = f(g(x) + b1(x) + b2(x) + b3(x))
        #where f() is a linear function and each bank has a corresponding a_value
    prediction = pd.DataFrame(a_values[0]*X["bcs"] + a_values[1]*X["lyg"] + a_values[2]*X["hsbc"] + a_values[3]*X["rbs"])
    prediction.columns = ["bcs"]
    #calculate the mse of the predictions ignoring the first as it was not predicted
    mse = np.mean((prediction - y)**2)
    mse = mse[0]
    return prediction,mse

#define a function for changing the a_values
def refine(y,X,a_values,roc,to_change):
    #input the target and supporting dataframes
    #input a_values and the rate of change of the a_values
    #to_change is the a_value that is to be changed
    a_values_main = a_values.copy() # create a copy to find when to stop iterating

    test = 0 # iterate until a condition is met
    while test == 0:
        target,mse = predict(y,X,a_values_main)
        #find the mse for the input a_values
        
        a_values[to_change] = a_values_main[to_change] + roc
        target_1,mse_1 = predict(y,X,a_values)
        #find the mse for the a_values with the specified a_value increased by roc
        
        a_values[to_change] = a_values_main[to_change] - roc
        target_2,mse_2 = predict(y,X,a_values)
      #find the mse for the a_values with the specified a_value decreased by roc
        if mse_1 < mse:
            a_values_main[to_change] = np.round(a_values_main[to_change]+roc,5)
            #if adding roc gave a better mse then replace the a_value with a_value + roc
        if mse_2 < mse:
            a_values_main[to_change] = np.round(a_values_main[to_change]-roc,5)
            #the same as above but if a decreased a_value is better
        else:
            test = 1
            #if the original a_value is the best then end the loop

    return a_values_main

a_values = [1,1,1,1]
#setup a_values by iteration or by guessing
learning_rate = 0.1

test_1 = 0
#loop until a condition is met
while test_1 == 0:
    a_values_orig = a_values.copy() # create a copy of a_values for comparisson

    #loop over the function to update each a_value to the best one
    a_values = refine(y,X,a_values,learning_rate,0)
    a_values = refine(y,X,a_values,learning_rate,1)
    a_values = refine(y,X,a_values,learning_rate,2)
    a_values = refine(y,X,a_values,learning_rate,3)

    #if the a_values after the loop is the same as before then stop the loop
    if a_values_orig == a_values:
        test_1 = 1
print(a_values)
#plot of original data and predicted data by linear function
prediction,mse = predict(y,X,a_values)
data = pd.concat([y,prediction],axis = 1)
data.columns = ["target","prediction"]
a = sns.lineplot(data = data)
a.set_title("Plot of BCS data and predicted data by linear function")
plt.show()

#plot of original data and predicted data by each day is equal to previous day

#set the predicted value equal to the previous days value
prediction_2 = pd.DataFrame(X["bcs"])

mse_2 = np.mean((prediction_2 - y)**2)
data_2 = pd.concat([y,prediction_2],axis = 1)
data_2.columns = ["target","prediction"]
a = sns.lineplot(data = data_2)
a.set_title("Plot of BCS data and predicted data by equal to previous day")
plt.show()

print(mse,mse_2) # show difference between mse of both functions

#%%analysis of linear regression function
#setup the a_values to use

prediction,mse = predict(y,X,a_values)
#plot of bcs data against predicted data using a_values from previously
data = pd.concat([y,prediction],axis = 1)
data.columns = ["target","prediction"]

a = sns.regplot(data = data,x = "prediction",y = "target")
a.set_ylim([-15,5])
a.set_xlim([-15,5])
a.set_title("Regression plot of bcs data against linear prediction data")
plt.show()

#plot of residuals
a = sns.scatterplot(data = data["target"] - data["prediction"])
a.set_title("plot of residuals of linearly predicted data")
plt.show()

#plot regression using the prediction equal to previous day
a = sns.regplot(data = data_2,x = "prediction",y = "target")
a.set_ylim([-15,5])
a.set_xlim([-15,5])
a.set_title("Regression plot of bcs data against previous day prediction")
plt.show()

a = sns.scatterplot(data = data_2["target"] - data_2["prediction"])
a.set_title("plot of residuals of previous day prediction")
plt.show()

#the linear prediction plot gives not the best predictions but it is much better than
#predicting by previous day. 
#this just shows that a linear function is not optimal for finding an output
#and a non-linear function would be better possibly


#%%Adaline

#using y and X from previously


a_values = [1,1,1,1] # guess a_values
learning_rate = 0.0001 # choose a learning rate
prediction,mse = predict(y,X,a_values) # find an intial prediction and mse
test = 0
mse_list = [] # create an empty list of mse
mse_list.append(mse)  # add in first mse
j = 0
while test == 0:
    j = j+1 
    for i in range(len(a_values)): # loop over the a_values
        column = X.columns[i] # find the column to use
        #apply formula for updating weights of adaline
        a_values[i] =a_values[i] + learning_rate * (pd.DataFrame(X[column]).T.dot(y - prediction)).bcs[0]
    
    #find mse using new a_values
    prediction,mse = predict(y,X,a_values)
    mse_list.append(mse) # add new mse to the list
    if mse_list[j] == mse_list[j-1]: # check if mse is equal to previous mse
        #this will be limited by how many significant figures are stored by computer
        test = 1
        
#plot graph of mse over time
a = sns.lineplot(data = mse_list)
a.set_title("value of MSE over time")
a.set_ylabel("MSE")
a.set_xlabel("Iteration")
plt.plot()