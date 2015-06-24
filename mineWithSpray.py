'''
Created on Jun 8, 2015

@author: ebrahimi
'''
'''
Created on Jun 1, 2015

@author: ebrahimi
'''
import pandas as pd
import numpy as np
from sklearn import ensemble, preprocessing

# Load dataset 
train = pd.read_csv('C:/home/ebrahimi/Kaggle/WNV/train.csv')
test = pd.read_csv('C:/home/ebrahimi/Kaggle/WNV/test.csv')
sample = pd.read_csv('C:/home/ebrahimi/Kaggle/WNV/sampleSubmission.csv')
weather = pd.read_csv('C:\home\ebrahimi\Kaggle\WNV\weather.csv')
spray = pd.read_csv('C:\home\ebrahimi\Kaggle\WNV\spray.csv');

# Get labels
labels = train.WnvPresent.values

# replace some missing values and T with -1
weather = weather.replace('M', -100)
weather = weather.replace('-', -100)
weather = weather.replace('T', -100)
weather = weather.replace(' T', -100)
weather = weather.replace('  T', -100)
## replace missing codesums
weather = weather.replace(' ', -100)
weather = weather.drop(['SnowFall', 'Depth'], axis=1);

# Split station 1 and 2 and join horizontally
weather_stn1 = weather[weather['Station']==1]
weather_stn2 = weather[weather['Station']==2]
weather_stn1 = weather_stn1.drop('Station', axis=1)
weather_stn2 = weather_stn2.drop('Station', axis=1)
weather = weather_stn1.merge(weather_stn2, on='Date')

for index, row in weather.iterrows():    
    
    if row['Tmax_x'] ==-1 and row['Tmax_y']!=-1: weather.ix[index,'Tmax_x'] = row['Tmax_y']    
    
    if row['Tmin_x'] ==-1 and row['Tmin_y']!=-1: weather.ix[index,'Tmin_x'] = row['Tmin_y']    
    
    if row['Tavg_x'] ==-1 and row['Tavg_y']!=-1: weather.ix[index,'Tavg_x'] = row['Tavg_y']    
    
    if row['Depart_x'] ==-1 and row['Depart_y']!=-1: weather.ix[index,'Depart_x'] = row['Depart_y']
        
    if row['DewPoint_x'] ==-1 and row['DewPoint_y']!=-1: weather.ix[index,'DewPoint_x'] = row['DewPoint_y']    
    
    if row['WetBulb_x'] ==-1 and row['WetBulb_y']!=-1: weather.ix[index,'WetBulb_x'] = row['WetBulb_y']  
    
    if row['Heat_x'] ==-1 and row['Heat_y']!=-1: weather.ix[index,'Heat_x'] = row['Heat_y']
    
    if row['Cool_x'] ==-1 and row['Cool_y']!=-1: weather.ix[index,'Cool_x'] = row['Cool_y']
    
    if row['Sunrise_x'] ==-1 and row['Sunrise_y']!=-1: weather.ix[index,'Sunrise_x'] = row['Sunrise_y']
    
    if row['Sunset_x'] ==-1 and row['Sunset_y']!=-1: weather.ix[index,'Sunset_x'] = row['Sunset_y']
                
    if row['CodeSum_x'] ==-1 and row['CodeSum_y']!=-1: weather.ix[index,'CodeSum_x'] = row['CodeSum_y']    
    
    if row['PrecipTotal_x']!=-1 and row['PrecipTotal_y']!=-1: weather.ix[index,'PrecipTotal_x'] = (float(row['PrecipTotal_x'])+float(row['PrecipTotal_y']))/2
    elif row['PrecipTotal_x'] ==-1 and row['PrecipTotal_y']!=-1: weather.ix[index,'PrecipTotal_x'] = row['PrecipTotal_y']   

    if row['StnPressure_x']!=-1 and row['StnPressure_y']!=-1: weather.ix[index,'StnPressure_x'] = (float(row['StnPressure_x'])+float(row['StnPressure_y']))/2
    elif row['StnPressure_x'] ==-1 and row['StnPressure_y']!=-1: weather.ix[index,'StnPressure_x'] = row['StnPressure_y']   

    if row['SeaLevel_x']!=-1 and row['SeaLevel_y']!=-1: weather.ix[index,'SeaLevel_x'] = (float(row['SeaLevel_x'])+float(row['SeaLevel_y']))/2
    elif row['SeaLevel_x'] ==-1 and row['SeaLevel_y']!=-1: weather.ix[index,'SeaLevel_x'] = row['SeaLevel_y']   

    if row['ResultSpeed_x']!=-1 and row['ResultSpeed_y']!=-1: weather.ix[index,'ResultSpeed_x'] = (float(row['ResultSpeed_x'])+float(row['ResultSpeed_y']))/2
    elif row['ResultSpeed_x'] ==-1 and row['ResultSpeed_y']!=-1: weather.ix[index,'ResultSpeed_x'] = row['ResultSpeed_y']
    
    if row['ResultDir_x'] ==-1 and row['ResultDir_y']!=-1: weather.ix[index,'ResultDir_x'] = row['ResultDir_y']

    if row['AvgSpeed_x']!=-1 and row['AvgSpeed_y']!=-1: weather.ix[index,'AvgSpeed_x'] = (float(row['AvgSpeed_x'])+float(row['AvgSpeed_y']))/2
    elif row['AvgSpeed_x'] ==-1 and row['AvgSpeed_y']!=-1: weather.ix[index,'AvgSpeed_x'] = row['AvgSpeed_y']  
 
##drop y columns 
weather = weather.drop(['Tmax_y', 'Tmin_y', 'Tavg_y','Depart_y','DewPoint_y','WetBulb_y','Heat_y','Cool_y','Sunrise_y','Sunset_y','CodeSum_y','PrecipTotal_y','StnPressure_y','SeaLevel_y','ResultSpeed_y','ResultDir_y','AvgSpeed_y'], axis =1)     
    
weather.to_csv('C:/home/ebrahimi/Kaggle/WNV/weather_processed.csv', index=False);

# Functions to extract month and day from dataset
# You can also use parse_dates of Pandas.
def create_month(x):
    return x.split('-')[1]

def create_day(x):
    return x.split('-')[2]

# def create_year(x):
#     return x.split('-')[0]

train['month'] = train.Date.apply(create_month)
train['day'] = train.Date.apply(create_day)
# train['year'] = train.Date.apply(create_year);
test['month'] = test.Date.apply(create_month)
test['day'] = test.Date.apply(create_day)
# test['year'] = test.Date.apply(create_year);

# Add integer latitude/longitude columns
train['Lat_3dig'] = train.Latitude.apply(lambda x: round(x, 3))
train['Long_3dig'] = train.Longitude.apply(lambda x: round(x, 3))
test['Lat_3dig'] = test.Latitude.apply(lambda x: round(x, 3))
test['Long_3dig'] = test.Longitude.apply(lambda x: round(x, 3))
spray['Lat_3dig'] = spray.Latitude.apply(lambda x: round(x, 3))
spray['Long_3dig'] = spray.Longitude.apply(lambda x: round(x, 3))


# drop address columns
train = train.drop(['Address', 'AddressNumberAndStreet', 'Block','Street', 'AddressAccuracy', 'day', 'NumMosquitos', 'Latitude', 'Longitude'], axis = 1)
test = test.drop(['Address', 'AddressNumberAndStreet', 'Block','Street', 'AddressAccuracy', 'day', 'Latitude', 'Longitude'], axis = 1)
spray = spray.drop(['Longitude', 'Date', 'Time'] , axis = 1) #Latitude is not removed deliberately

spray['Latitude'] = spray.Latitude.apply(int); # just used as a boolean to identify the sprayed rows
# Merge with weather data
train = train.merge(weather, on='Date')
test = test.merge(weather, on='Date')
train = train.drop(['Date'], axis = 1)
test = test.drop(['Date'], axis = 1)

###Merge with Spray data
train = pd.merge(train, spray, how='left', on=['Lat_3dig','Long_3dig'])
test = pd.merge(test, spray, how='left', on=['Lat_3dig','Long_3dig'])
# 'Latitude_x', 'Longitude_x', 'Latitude_y', 'Latitude_y' 
# Convert categorical data to numbers
lbl = preprocessing.LabelEncoder()
lbl.fit(list(train['Species'].values) + list(test['Species'].values))
train['Species'] = lbl.transform(train['Species'].values)
test['Species'] = lbl.transform(test['Species'].values)

lbl.fit(list(train['Trap'].values) + list(test['Trap'].values))
train['Trap'] = lbl.transform(train['Trap'].values)
test['Trap'] = lbl.transform(test['Trap'].values)

# drop columns with -1s
train = train.ix[:,(train != -100).any(axis=0)]
test = test.ix[:,(test != -100).any(axis=0)]

print("hoooo")

train.to_csv('C:/home/ebrahimi/Kaggle/WNV/train_processed_New_3DigMissing-100.csv', index=False)
test.to_csv('C:/home/ebrahimi/Kaggle/WNV/test_processed_New_3DigMissing-100.csv', index=False);

