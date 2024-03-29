import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from matplotlib.pyplot import rcParams
rcParams['figure.figsize']=10,6
from datetime import datetime
#reading daily_data
daily_data = pd.read_csv('/kaggle/input/temperature-data-albany-new-york/daily_data.csv')
daily_data.head(10)
#reading hourly_data
hourly_data = pd.read_csv('/kaggle/input/temperature-data-albany-new-york/hourly_data.csv')
hourly_data.head(10)
#reading monthly_data
monthly_data = pd.read_csv('/kaggle/input/temperature-data-albany-new-york/monthly_data.csv')
monthly_data.head(10)
#reading three_hour_data
three_hour_data = pd.read_csv('/kaggle/input/temperature-data-albany-new-york/three_hour_data.csv')
three_hour_data.head(10)
daily_data.describe()
#columns in daily_data
print(daily_data.columns)
#finding if there is any null value
daily_data.isnull().sum()
#finding the unique_values
unique_values=daily_data.nunique()
print(unique_values)
#replacing the missing values
daily_data.dropna(inplace=True)
# Check for duplicates
print(daily_data.duplicated().sum())
daily_data.head()
from datetime import datetime
daily_data['DATE'] = pd.to_datetime(daily_data['DATE'] , infer_datetime_format = True)
n_daily_data = daily_data
n_daily_data.head(5)
grouped_col = ['DailyAverageDryBulbTemperature' , 'DailyPrecipitation' , 'DailyHeatingDegreeDays']
selected_data= n_daily_data[grouped_col]
des_state = selected_data.describe()
print(des_state) 
# Convert 'DATE' column to datetime format

daily_data['DATE']=pd.to_datetime(daily_data['DATE'])
plt.figure(figsize=(10,6))

#finding the daily avg dry bulb temperature
plt.plot(daily_data['DATE'], daily_data['DailyAverageDryBulbTemperature'] , color='green')
plt.title('Daily Average Dry Bulb Temperature')
plt.xlabel('Date')
plt.ylabel('frequency')
plt.xticks(rotation=45)
plt.show()



#finding the daily precipitation
plt.figure(figsize=(12,8))
plt.plot(n_daily_data['DATE'], n_daily_data['DailyPrecipitation'] , color='green')
plt.title('Daily Average precipitation')
plt.xlabel('Date')
plt.ylabel('Precipitation (inches)')
plt.ylim(0, 35)
plt.grid(True)
plt.show()

#finding the daily avg wind speed
plt.figure(figsize=(10,6))
plt.plot(n_daily_data['DATE'] , n_daily_data['DailyAverageWindSpeed'],color='orange')
plt.title('Time series plot for Daily average wind speed')
plt.xlabel('Date')
plt.ylabel('wind speed ')
plt.grid(True)
plt.show()

daily_data['Month']=daily_data['DATE'].dt.month
daily_data['Year']=daily_data['DATE'].dt.year
#monthly variation of average temperature
avgtemp = daily_data.groupby(['Month'])['DailyAverageDryBulbTemperature'].mean()
plt.figure(figsize=(10,6))
plt.plot(avgtemp.index , avgtemp.values , marker ='*' , linestyle='-')
plt.title('Monthly variation of average temperature')
plt.xlabel('Month')
plt.ylabel('avg temp')
plt.xticks(range(1,13),['JAN' , 'FEB','MAR','APR','MAY','JUNE','JULY','AUG','SEP','OCT','NOV','DEC'])
plt.grid(True)
plt.show()
#finding the  heating degree days
plt.figure(figsize=(10,6))
plt.plot(daily_data['DATE'], daily_data['DailyHeatingDegreeDays'] , color='red')
plt.title('Daily Heating Degree Days ')
plt.xlabel('Date')
plt.ylabel('heat degree days')
plt.xticks(rotation=45)
plt.show()

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(n_daily_data['DailyAverageDryBulbTemperature'],model='additive',period = 30)
decomposition.plot()
plt.show()
