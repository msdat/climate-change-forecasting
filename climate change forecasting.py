#!/usr/bin/env python
# coding: utf-8

# #### Climate Change Forecasting Using Deep Learning

# In[1]:


# import key libraries
import pandas as pd
import plotly.express as px
from copy import copy
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
import plotly.offline as py
import plotly.express as px
import plotly.graph_objects as go


# In[2]:


from jupyterthemes import jtplot
jtplot.style(theme = 'monokai', context = 'notebook', ticks = True, grid = False) 


# In[3]:


# Data Source: https://www.kaggle.com/berkeleyearth/climate-change-earth-surface-temperature-data/notebooks?datasetId=29&sortBy=voteCount


# In[4]:


# Read files
temperature_df = pd.read_csv('GlobalLandTemperaturesByCountry.csv')


# In[ ]:


temperature_df


# In[ ]:


temperature_df.shape


# In[ ]:


temperature_df.describe()


# ### Exploratory data analysis 

# In[ ]:


# Check the unique countries
temperature_df['Country'].unique()


# In[ ]:


# Check for missing values
temperature_df.isnull().sum()


# In[ ]:


# Check the dataframe information
temperature_df.info()


# ### data cleaning 

# In[9]:


# Do groupby country to see the count of values available for each country
country_group_df = temperature_df.groupby(by = 'Country').count().reset_index('Country').rename(columns={'AverageTemperature':'AverageTemperatureCount','AverageTemperatureUncertainty' : 'AverageTemperatureUncertaintyCount'})


# In[ ]:


country_group_df


# In[ ]:


country_group_df['Country']


# In[12]:


import plotly.express as px
fig = px.bar(country_group_df, x = 'Country', y = 'AverageTemperatureCount')
fig.show()


# In[13]:


fig = px.bar(country_group_df, x = 'Country', y = 'AverageTemperatureUncertaintyCount')
fig.show()


# In[14]:


# Plot histogram
fig = px.histogram(country_group_df, x = "AverageTemperatureCount")
fig.show()


# In[15]:


fig = px.histogram(country_group_df, x = "AverageTemperatureUncertaintyCount")
fig.show()


# In[ ]:


country_group_df[(country_group_df['AverageTemperatureCount'] < 1500) | (country_group_df['AverageTemperatureUncertaintyCount'] < 1500)]


# In[17]:


# Find countries with less than 1500 data info
countries_with_less_data = country_group_df[(country_group_df['AverageTemperatureCount'] < 1500) | (country_group_df['AverageTemperatureUncertaintyCount'] < 1500)].index.tolist()


# In[18]:


countries_with_less_data


# In[ ]:


~temperature_df['Country'].isin(countries_with_less_data)


# In[20]:


# Remove countries with less data info
temperature_df = temperature_df[~temperature_df['Country'].isin(countries_with_less_data)]


# In[21]:


temperature_df.reset_index(inplace = True, drop = True)


# In[ ]:


temperature_df


# In[23]:


# Fill missing values by doing rolling average on past 730 days
temperature_df['AverageTemperature'] = temperature_df['AverageTemperature'].fillna(temperature_df['AverageTemperature'].rolling(730, min_periods = 1).mean())


# In[24]:


# Fill missing values by doing rolling average on past 730 days
temperature_df['AverageTemperatureUncertainty']= temperature_df['AverageTemperatureUncertainty'].fillna(temperature_df['AverageTemperatureUncertainty'].rolling(730, min_periods=1).mean())


# In[ ]:


temperature_df.isna().sum()


# In[ ]:


temperature_df['Country'].unique()


# In[ ]:


duplicates = []
for i in temperature_df['Country'].unique():
    if '(' in i:
        duplicates.append(i)
duplicates


# In[28]:


# replace duplicates
temperature_df = temperature_df.replace(duplicates, ['Congo', 'Denmark','Falkland Islands','France','Netherlands','United Kingdom'])


# In[ ]:


temperature_df['Country'].unique()


# ### data visualization 
# 

# In[ ]:


countries = temperature_df['Country'].unique().tolist()
countries


# In[31]:


# mean temperature for each country
mean_temperature = []
for i in countries:
    mean_temperature.append(temperature_df[temperature_df['Country'] == i]['AverageTemperature'].mean())


# In[32]:


# Plot mean teamperature of countries
data = [ dict(type = 'choropleth', # type of map
              locations = countries, # location names
              z = mean_temperature, # temperature of countries
              locationmode = 'country names')
       ]

layout = dict(title = 'Average Global Land Temperatures',
              geo = dict(showframe = False,
                         showocean = True, # to show the ocean
                         oceancolor = 'aqua',
                         projection = dict(type = 'orthographic'))) # to get the globe view),

fig = dict(data = data, layout = layout)
py.iplot(fig, validate = False, filename = 'worldmap')


# In[ ]:


# year of recorded data for visualization 
temperature_df['year'] = temperature_df['dt'].apply(lambda x: x.split('-')[0])
temperature_df


# In[34]:


# creating the animation to see the global temperature change
fig = px.choropleth(temperature_df, locations = 'Country',
                    locationmode = 'country names', # locations 
                    color = 'AverageTemperature', # column representing the temperature
                    hover_name = "Country", # column to add to hover information
                    animation_frame = 'year', # timeframe for animation
                    color_continuous_scale = px.colors.sequential.deep_r)
# py.plot(fig)
fig.show()


# In[35]:


# To get  global average tempeature over years
df_global = temperature_df.groupby('year').mean().reset_index()


# In[36]:


df_global['year'] = df_global['year'].apply(lambda x: int(x))
df_global = df_global[df_global['year'] > 1850]


# In[37]:


# uncertainity upper bound 
trace1 = go.Scatter(
    x = df_global['year'], 
    y = np.array(df_global['AverageTemperature']) + np.array(df_global['AverageTemperatureUncertainty']), # Adding uncertinity
    name = 'Uncertainty top',
    line = dict(color = 'green'))

# uncertainity lower bound
trace2 = go.Scatter(
    x = df_global['year'] , 
    y = np.array(df_global['AverageTemperature']) - np.array(df_global['AverageTemperatureUncertainty']), # Subtracting uncertinity
    fill = 'tonexty',
    name = 'Uncertainty bottom',
    line = dict(color = 'green'))

# recorded temperature
trace3 = go.Scatter(
    x = df_global['year'] , 
    y = df_global['AverageTemperature'],
    name = 'Average Temperature',
    line = dict(color='red'))
data = [trace1, trace2, trace3]

layout = go.Layout(
    xaxis = dict(title = 'year'),
    yaxis = dict(title = 'Average Temperature, °C'),
    title = 'Average Land Temperatures Globally',
    showlegend = False)

fig = go.Figure(data = data, layout = layout)
py.iplot(fig)


# In[ ]:


# modeling for US data
US_df = temperature_df[temperature_df['Country'] == 'United States'].reset_index(drop = True)
US_df


# In[39]:


fig = px.line(title = 'US Temperature Data')
US_df_updated = US_df[US_df['year'] > '1813']
fig.add_scatter(x = US_df_updated['dt'], y = US_df_updated['AverageTemperature'], name = 'US Temperature')
fig.show()


# ### data preparation for model training 

# In[40]:


# Get the month of recording, to use as a feature
temperature_df['Month'] = temperature_df['dt'].apply(lambda x: int(x.split('-')[1]))


# In[41]:


# To get the global average tempeature over years
df_global_monthly = temperature_df.groupby(['dt']).mean().reset_index()


# In[ ]:


df_global_monthly


# In[43]:


# creating data for training the time series model
def prepare_data(df, feature_range):
    # Get the columns
    columns = df.columns
    # For the given range, create lagged input feature for the given columns
    for i in range(1, (feature_range + 1)):
        for j in columns[1:]:
            name = j + '_t-' + str(i)
            df[name] = df[j].shift((i))
    # Create the target by using next value as the target
    df['Target'] = df['AverageTemperature'].shift(-1)
    return df


# In[44]:


df_global_monthly = prepare_data(df_global_monthly, 3)


# In[ ]:


df_global_monthly


# In[46]:


df_global_monthly = df_global_monthly.dropna().reset_index(drop = True)


# In[ ]:


df_global_monthly


# In[48]:


train = df_global_monthly[:int(0.9 * len(df_global_monthly))].drop(columns = 'dt').values


# In[49]:


test = df_global_monthly[int(0.9 * len(df_global_monthly)):].drop(columns = 'dt').values


# In[50]:


scaler = MinMaxScaler(feature_range = (0, 1))
train  = scaler.fit_transform(train)
test   = scaler.transform(test)


# In[51]:


# split data into input features, targets
train_x, train_y = train[:,:-1], train[:,-1]
test_x, test_y = test[:,:-1], test[:,-1]


# In[52]:


# reshape input to 3D [samples, timesteps, features]
train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))
print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)


# ###  LSTM Model for predicting global temperature trend

# In[ ]:


def create_model(train_x):
    # create model
    inputs = keras.layers.Input(shape = (train_x.shape[1], train_x.shape[2]))
    x = keras.layers.LSTM(50,return_sequences =  True)(inputs)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.LSTM(50, return_sequences = True)(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.LSTM(50)(x)
    outputs = keras.layers.Dense(1, activation = 'linear')(x)

    model = keras.Model(inputs = inputs, outputs = outputs)
    model.compile(optimizer = 'adam', loss = "mse")
    return model


# In[54]:


model = create_model(train_x)
model.summary()


# In[ ]:


# fit the network
history = model.fit(train_x, train_y, epochs = 50, batch_size = 72, validation_data = (test_x, test_y), shuffle = False)


# In[ ]:


def plot_history(history):
    # plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.grid()
    plt.legend()
    plt.show()


# In[ ]:


plot_history(history)


# In[ ]:


# model performance 
def prediction(model,test_x,train_x, df):
    # Predict using the model
    predict =  model.predict(test_x)

    # Reshape test_x and train_x for visualization  and inverse-scaling purpose
    test_x = test_x.reshape((test_x.shape[0], test_x.shape[2]))
    train_x = train_x.reshape((train_x.shape[0], train_x.shape[2]))

    # Concatenate test_x with predicted value
    predict_ = np.concatenate((test_x, predict),axis = 1)

    predict_ = scaler.inverse_transform(predict_)
    original_ = scaler.inverse_transform(test)

    # Create dataframe to store the predicted and original values
    pred = pd.DataFrame()
    pred['dt'] = df['dt'][-test_x.shape[0]:]
    pred['Original'] = original_[:,-1]
    pred['Predicted'] = predict_[:,-1]

    pred['Error'] = pred['Original'] - pred['Predicted']
    
    # Create dataframe for visualization
    df = df[['dt','AverageTemperature']][:-test_x.shape[0]]
    df.columns = ['dt','Original']
    original = df.append(pred[['dt','Original']])
    df.columns = ['dt','Predicted']
    predicted = df.append(pred[['dt','Predicted']])
    original = original.merge(predicted, left_on = 'dt',right_on = 'dt')
    return pred, original


# In[ ]:


pred, original = prediction(model, test_x, train_x, df_global_monthly )


# In[ ]:


def plot_error(df):

    # Plotting the Current and Predicted values
    fig = px.line(title = 'Prediction vs. Actual')
    fig.add_scatter(x = df['dt'], y = df['Original'], name = 'Original', opacity = 0.7)
    fig.add_scatter(x = df['dt'], y = df['Predicted'], name = 'Predicted', opacity = 0.5)
    fig.show()

    fig = px.line(title = 'Error')
    fig = fig.add_scatter(x = df['dt'], y = df['Error'])
    fig.show()


# In[ ]:


def plot(df):
    # Plotting the Current and Predicted values
    fig = px.line(title = 'Prediction vs. Actual')
    fig.add_scatter(x = df['dt'], y = df['Original'], name = 'Original', opacity = 0.7)
    fig.add_scatter(x = df['dt'], y = df['Predicted'], name = 'Predicted', opacity = 0.5)
    fig.show()


# In[ ]:


plot(original)


# In[ ]:


plot_error(pred)


# ### data prep for US data

# In[ ]:


US_df = temperature_df[temperature_df['Country'] == 'United States'].reset_index(drop = True)
US_df


# In[ ]:


US_df = US_df.drop(['Country', 'year'], axis = 1)


# In[ ]:


# training data
US_df = prepare_data(US_df, 3)


# In[ ]:


US_df


# In[ ]:


US_df = US_df.dropna().reset_index(drop = True)


# In[ ]:


train = US_df[:int(0.9 * len(US_df))].drop(columns = 'dt').values


# In[ ]:


test = US_df[int(0.9 * len(US_df)):].drop(columns = 'dt').values


# In[ ]:


train.shape


# In[ ]:


scaler = MinMaxScaler(feature_range = (0, 1))
train  = scaler.fit_transform(train)
test   = scaler.transform(test)


# In[ ]:


# split data into input features and targets
train_x, train_y = train[:,:-1], train[:,-1]
test_x, test_y = test[:,:-1], test[:,-1]


# In[ ]:


# reshape input to 3D 
train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))
print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)


# In[ ]:


train_x


# ###  LSTM Model for predicting US temperature trend

# In[ ]:


model = create_model(train_x)
model.summary()


# In[ ]:


# fit the network
history = model.fit(train_x, train_y, epochs = 50, batch_size = 72, validation_data = (test_x, test_y), shuffle = False)


# In[ ]:


plot_history(history)


# In[ ]:


# us model performance 
pred, original = prediction(model, test_x, train_x, US_df )


# In[ ]:


plot(original)


# In[ ]:


plot_error(pred)


# In[ ]:




