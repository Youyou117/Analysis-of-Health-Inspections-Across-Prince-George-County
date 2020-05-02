#!/usr/bin/env python
# coding: utf-8

# ## Analysis of Health Inspections Across Prince George's County

# The data for this assignment comes from the Prince George's County Health Department. Details about the data are located at: https://data.princegeorgescountymd.gov/Health/Food-Inspection/umjn-t2iz. The data set contains almost 35,000 rows across 29 columns.

# ### Data Preparation and Cleansing

# Read in the data.

# In[1]:


#import the dataset
import pandas as pd
df=pd.read_csv("HW7_Food Inspection.csv")
df.head()


# Take a look at the table of the unique values of Category

# In[2]:


#get the unique values of Category
df['Category'].value_counts()


# Which values do you believe represent a restaurant? 
# 
# 

# Values such as 'Carry-out','Fast Food','Full Service','Bakery/Catering','Restaurant','Catering Only', 'Cafeteria', 'Snack Bar/Concession Stand', 'Pre-Packaged Only', 'Deli','Seafood','Fast Food - Chain', 'Full Service/Catering','Buffet','Pizza','Fast Food - Local','Delivery Only', '@Full Service', 'Diner' are considered restaurant.

# Create a single dummy variable for restaurant that combines multiple values from Category.

# In[3]:


#Create a single dummy variable for restaurant that combines multiple values from Category.
import numpy as np
restaurant=[ 'Carry-out','Fast Food','Full Service','Bakery/Catering','Restaurant','Catering Only', 'Cafeteria', 'Snack Bar/Concession Stand', 'Pre-Packaged Only', 'Deli','Seafood','Fast Food - Chain', 'Full Service/Catering','Buffet','Pizza','Fast Food - Local','Delivery Only', '@Full Service', 'Diner']
df['Restaurant_or_not']=np.where(df['Category'].isin(restaurant),1,0)
df.head()


# Convert the Inspection_date column into a datetime column and create new date columns.

# In[4]:


#Convert the Inspection_date column into a datetime column.
import datetime as dt
get_ipython().run_line_magic('timeit', "pd.to_datetime(df['Inspection_date'], infer_datetime_format=True)")


# In[5]:


#Create a new column for the year of the inspection. 
df['Year'] = pd.DatetimeIndex(df['Inspection_date']).year
df.head()


# In[6]:


#Create a new column for the month of the inspection.
df['Month'] = pd.DatetimeIndex(df['Inspection_date']).month
df.head()


# In[7]:


#Create a column for the year and month.
df['Month_Year'] = pd.to_datetime(df['Inspection_date']).dt.to_period('M')
df.head()


# For each column with the type of compliance, e.g. "Rodent and Insects," create a dummy variable that is 1 if the establishment is out of compliance and 0 otherwise.

# In[8]:


#find out columns with the type of compliance
df3=df[df.isin(["In Compliance"])|df.isin(["Out of Compliance"])].dropna(how='all',axis=1)
df3.head()


# In[9]:


# list all columns with the type of compliance
columns=list(df3.columns)  
columns


# In[10]:


#check whether columns have null values
df3.isnull().sum()


# From the result we see no column with the type of compliance contains null values

# In[11]:


#change the values of each raw to dummies
for col in columns:
    df[col]=df[col].astype(str)
    df[col].replace("In Compliance",0,inplace=True)
    df[col].replace("Out of Compliance",1,inplace=True)


# In[12]:


df[columns].head()


# Create a new column that contains the number of violations for that inspection (the number of categories where the establishment was not in compliance). 

# In[13]:


#sum all the values in columns
df["Violation_counts"]=df[columns].astype(int).sum(axis=1)
df.head()


# Create a dummy variable that is 1 if the establishment is out of compliance in any category.

# In[14]:


#Create a dummy variable 
df["Compliance_or_not"]=np.where(df['Violation_counts']==0,0,1)
df.head()


# For establishments with multiple inspections, create a new DataFrame in wide format. Keep only the establishment ID, Category, Inspection_date, and number of violations.

# In[15]:


#create a new DataFrame in wide format
newdf=df[['Establishment_id', 'Category', 'Inspection_date','Violation_counts']]
newdf.head()


# Make sure category is consistent within ID and resolve any discrepancies if necessary (i.e. each establishment has only one category). 

# In[16]:


#find the unique values of 'Establishment_id'
id=newdf['Establishment_id'].unique()
id=pd.Series(id)


# In[17]:


#find 'Establishment_id' which has more than 1 category
uid=[]
for i in id:
    counts=np.count_nonzero(newdf[newdf['Establishment_id']==i]['Category'].unique())
    if counts>1:
        uid.append(i)
uid=pd.Series(uid)


# In[18]:


#resolve any discrepancies 
#only leave the first category for each 'Establishment_id' that has more than 1 category
df4=pd.DataFrame()
for u in uid:
    choose=newdf[newdf['Establishment_id']==u]['Category'].unique()[0] 
    value=newdf[newdf['Establishment_id']==u][newdf['Category']==choose]
    df4=df4.append(value,ignore_index=True)
df4.head()


# In[19]:


#create a dataframe whose 'Establishment_id' only has no more than 1 category
df5=pd.DataFrame()
for o in id:
    if o not in uid:
        value2=newdf[newdf['Establishment_id']==o]
        df5=df5.append(value2,ignore_index=True)
df5.head()


# In[20]:


#contanete two dataframes
#this is a dataframe whose category is consistent within ID  
newnewdf= pd.concat([df4, df5], axis=0,ignore_index=True)
newnewdf.sort_values(by='Establishment_id').head()


# Reshape from long to wide (pivot) such that each establishment is a row and you have a column for the date and number of violations for inspection 1, inspection 2, inspection 3, etc.

# In[21]:


tables=newdf.pivot_table(values='Violation_counts',index=['Establishment_id','Inspection_date'],aggfunc='sum',margins=True)
tables


# ## Statistics/Data Grouping

# What is the most common type of violation? The compliance categories are not mutually exclusive because one restaurant can have multiple violations. Create a table with the number of violations by violation type. Sort the table from the most common to least common violations.

# In[48]:


#count the umber of violations by violation type
dfc=df[columns]
counts=dfc[dfc == 1].sum(axis=0).sort_values(ascending=False)
counts


# The most common type of violation is Cold_holding_temperature.

# ## Data Visualization

# Limit the analysis to restaurants using the dummy variable indicator created previously.

# Create a bar graph showing the results of violations from Step 2a.

# In[49]:


#create a dataframe that only contain restaurants
dfr=df[df['Restaurant_or_not']==1]
#count the umber of violations by violation type for every restaurant
dfrr=dfr[columns]
countsr=dfrr[dfrr == 1].sum(axis=0).sort_values(ascending=False)
countsr


# In[50]:


#plot the bar graph
import seaborn as sns

ax = countsr.plot(kind='barh', rot=0,figsize=(10,8))                                  
ax.set_title("The number of violations for each violation type",fontdict={'fontsize': 20, 'fontweight': 700, 'color': 'maroon'}, pad=20)
ax.set_xlabel("Violation counts", fontsize=16)
ax.set_ylabel("Violation types", fontsize=16);

# set individual bar lables
for i in ax.patches:
    ax.text(i.get_width(), i.get_y(),             str(round((i.get_width()))), fontsize=12, color='blue')


# Create a line graph that shows the percent of restaurant inspections that have at least one violation by month and year.

# In[51]:


#find all restaurant inspections that have at least one violation
df10=dfr[dfr["Violation_counts"]!=0]


# In[52]:


#calculate the frequencies
grouped=df10.groupby(by='Year')["Violation_counts"].agg('count')
grouped2=dfr.groupby(by='Year')["Violation_counts"].agg('count')
frequency=grouped/grouped2
frequency


# In[53]:


year=pd.Series(range(2011,2023,1))


# In[54]:


#plot the yearly graph
import matplotlib.pyplot as plt 
plt.xlabel('Year')
plt.ylabel('Violation rate')
plt.title('Yearly Violation rate', fontdict={'fontsize': 20, 'fontweight': 700, 'color': 'maroon'}, pad=20)
plt.plot(frequency)
plt.xticks(year) 


# In[55]:


#calculate the frequencies
grouped3=df10.groupby(by='Month')["Violation_counts"].agg('count')
grouped4=dfr.groupby(by='Month')["Violation_counts"].agg('count')
frequency2=grouped3/grouped4
frequency2


# In[56]:


month=pd.Series(range(1,13,1))


# In[57]:


#plot the monthly graph
import matplotlib.pyplot as plt 
plt.xlabel('Month')
plt.ylabel('Violation rate')
plt.title('Monthly Violation rate', fontdict={'fontsize': 20, 'fontweight': 700, 'color': 'maroon'}, pad=20)
plt.plot(frequency2)
plt.xticks(month) 


# Are inspections getting harder or easier over time? Is there a particular month where more restaurants pass? 
# 
# 
# From the year graph we can see inspections got harder from 2011 to 2013, and then started to get easier unill 2017,and then after 2017 it got harder again.From the month graph we can see overall inspections get easier for the first half of the year,and then get harder for the second half of the year. Around May and June,more restaurants pass.

# Create a map that shows all of the restaurants. Color the restaurants with at least one violation in red.

# In[58]:


#extract longitude and latitude for all restaurants
dfr['LL']=dfr['Location'].str.extract('(\d+.\d+ \d+.\d+)')
dfr['LL'].head()


# In[59]:


dfr['Longitude']=dfr['LL'].str.split(' ').str[0]
dfr['Longitude'].head()


# In[60]:


dfr['Latitude']=dfr['LL'].str.split(' ').str[1]
dfr['Latitude'].head()


# In[61]:


#extract longitude and latitude for restaurant inspections that have at least one violation
df10['LL']=df10['Location'].str.extract('(\d+.\d+ \d+.\d+)')
df10['Longitude']=df10['LL'].str.split(' ').str[0]
df10['Latitude']=df10['LL'].str.split(' ').str[1]


# In[62]:


# Enter your OWN mapbox token here!
mapbox_access_token =  'pk.eyJ1IjoiZXZhemhvdTExNyIsImEiOiJjazlnNjFkbXkwY284M2tvMnN1NWNreG5xIn0.q3Q1pKC8fZcAPrjaoGNyyg'


# In[63]:


#create a map
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scattermapbox(
        text = dfr['Name'],
        lon = dfr['Longitude'],
        lat = dfr['Latitude'],
        hoverinfo='text',
        mode = 'markers',
        marker = dict(
                    color = 'green',
                    symbol = 'circle',
                    size=8
                )
    ))

fig.add_trace(go.Scattermapbox(
        lon = df10['Longitude'],
        lat = df10['Latitude'],
        mode = 'markers',
        marker = dict(
                    color = 'red',
                    size=8          
                ),
        hoverinfo='none'
    ))

fig.update_layout(
        title = 'Restaurants',
        mapbox=go.layout.Mapbox(
            accesstoken=mapbox_access_token,
            zoom=1
        )
    )

fig.show()


# Are there particular areas with more violations? ? 

# In[64]:


df10.groupby(by=['City'])['Violation_counts'].agg('sum').sort_values(ascending=False).head()


# We can see LAUREL,HYATTSVILLE,CAPITOL HEIGHTS,BOWIE and COLLEGE PARK have more more violations.

# In[65]:


dfcity=df10[(df10['City']=='LAUREL')|(df10['City']=='HYATTSVILLE')|(df10['City']=='CAPITOL HEIGHTS')|(df10['City']=='BOWIE ')|(df10['City']==' COLLEGE PARK ')]
dfcity.head()


# In[66]:


#extract longitude and latitude for cities with more violations
dfcity['LL']=dfcity['Location'].str.extract('(\d+.\d+ \d+.\d+)')
dfcity['Longitude']=dfcity['LL'].str.split(' ').str[0]
dfcity['Latitude']=dfcity['LL'].str.split(' ').str[1]


# In[67]:


city_map_data = go.Scattermapbox(
        lon = dfcity['Longitude'],
        lat = dfcity['Latitude'],
        text = dfcity['Name'],
        hoverinfo='text',
        mode = 'markers',
        marker = dict(
                    color = 'orange',
                    symbol = 'circle',
                    opacity = .5
                )
)

city_map_layout = go.Layout(
        title = 'Cities with more violations',
        mapbox=go.layout.Mapbox(
            accesstoken=mapbox_access_token,
            zoom=1
        )
    )

city_map = go.Figure(data=city_map_data, layout=city_map_layout)
city_map.show()


# From the map we can see restaurants which has more violations are located at two main areas, And in these two areas restaurants are very close to each other.
