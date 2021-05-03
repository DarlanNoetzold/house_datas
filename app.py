import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster
import plotly.express as px
from datetime import datetime

st.set_page_config(layout='wide')

@st.cache(allow_output_mutation=True)
def get_data(path):
    data = pd.read_csv(path)

    return data

#get_data
path = 'kc_house_data.csv'
data = get_data(path)

#add new features
data['price_m2'] = data['price'] / data['sqft_lot']

#get geofile
url = 'http://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson'

#data overview
f_attribute = st.sidebar.multiselect('Enter columns', data.columns)
f_zipcode = st.sidebar.multiselect('Enter zipcode', data['zipcode'].unique())

st.title('Data Overview')

if(f_zipcode != []) and (f_attribute != []):
    data = data.loc[data['zipcode'].isin(f_zipcode), f_attribute]
elif(f_zipcode != []) and (f_attribute == []):
    data = data.loc[data['zipcode'].isin(f_zipcode), :]
elif(f_zipcode == []) and (f_attribute != []):
    data = data.loc[:, f_attribute]
else:
    data = data.copy();


#Average metrics
df1 = data[['id', 'zipcode']].groupby('zipcode').count().reset_index()
df2 = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
df3 = data[['sqft_living', 'zipcode']].groupby('zipcode').mean().reset_index()
df4 = data[['price_m2', 'zipcode']].groupby('zipcode').mean().reset_index()

st.dataframe(data)

c1, c2 = st.beta_columns((1, 1))
#merge
m1 = pd.merge(df1,df2, on='zipcode', how='inner')
m2 = pd.merge(m1,df3, on='zipcode', how='inner')
df = pd.merge(m2,df4, on='zipcode', how='inner')

df.columns = ['zipcode', 'TOTAL HOUSE', 'PRICE', 'SQRT LINVING', 'PRICE/M2']

c1.header('Average Values')
c1.dataframe(df, height=600)

#Statistic Descriptive
num_attributes = data.select_dtypes(include=['int64', 'float64'])
media = pd.DataFrame(num_attributes.apply(np.mean))
mediana = pd.DataFrame(num_attributes.apply(np.median))
std = pd.DataFrame(num_attributes.apply(np.std))

max_ = pd.DataFrame(num_attributes.apply(np.max))
min_ = pd.DataFrame(num_attributes.apply(np.min))

df1 = pd.concat([max_,min_,media,mediana,std],axis=1).reset_index()

df1.columns = ['attributes', 'max', 'min', 'mean', 'median', 'std']

c2.header('Discriptive Analysis')
c2.dataframe(df1, height=600)

#Region overview
st.title('Region Overview')

c1, c2 = st.beta_columns((1, 1))
c1.header('Portfolio Density')

df = data.sample(10)

#base map - Folium
density_map = folium.Map(location=[data['lat'].mean(), data['long'].mean()], default_zoom_start=15)

marker_cluster = MarkerCluster().add_to(density_map)

for name, row in df.iterrows():
    folium.Marker([row['lat'], row['long']], popup='Price R${0} on: {1}. Features: {2} sqft, {3} bedrooms, {4} bathrooms, year built: {5}'.format(row['price'], row['date'], row['sqft_living'], row['bedrooms'], row['bathrooms'], row['yr_built'])).add_to(marker_cluster)

with c1:
    folium_static(density_map)

#commercial tag
st.sidebar.title('Commercial Options')
st.title('Commercial Attributes')

# Average price per year
data['date'] = pd.to_datetime(data['date'])

#filters
min_year_built = int(data['yr_built'].min())
max_year_built = int(data['yr_built'].max())

st.sidebar.subheader('Select Max Year Built')
f_year_built = st.sidebar.slider('Year Built', min_year_built, max_year_built, min_year_built)

st.header('Avarage Price per Year Built')

#data select
df = data.loc[data['yr_built'] < f_year_built]
df = df[['yr_built', 'price']].groupby('yr_built').mean().reset_index()

#plot
fig = px.line(df, x= 'yr_built', y='price')
st.plotly_chart(fig, use_container_width=True)

# Average price per day
st.header('Avarage Price per Day')
st.sidebar.subheader('Select Max Date')

#filters

min_date = data['date'].min().to_pydatetime('%Y-%m-%d')
max_date = data['date'].max().to_pydatetime('%Y-%m-%d')

f_date = st.sidebar.slider('Date', min_date, max_date, min_date)

#data select
data['date'] = pd.to_datetime((data['date']))
df = data.loc[data['date'] < f_date]
df = df[['date', 'price']].groupby('date').mean().reset_index()

#plot
fig = px.line(df, x= 'date', y='price')
st.plotly_chart(fig, use_container_width=True)

#Histograma
st.header('Price Distribution')
st.sidebar.subheader('Select Max Price')

#filter
min_price = int(data['price'].min())
max_price = int(data['price'].max())
avg_price = int(data['price'].mean())

#data filtering
f_price = st.sidebar.slider('Price', min_price, max_price, avg_price)
df = data.loc[data['price'] < f_price]

#data plot
fig = px.histogram(df, x='price', nbins=50)
st.plotly_chart(fig, use_container_width=True)

#distribution tags
st.sidebar.title('Attribute Options')
st.title('House Attribuites')

#filters
f_bedrooms = st.sidebar.selectbox('Max number of bedrooms', sorted(set(data['bedrooms'].unique())))
f_bathrooms = st.sidebar.selectbox('Max number of bathrooms', sorted(set(data['bathrooms'].unique())))

c1,c2=st.beta_columns(2)
#House per bedrooms
c1.header('Houses per bedrooms')
df = data[data['bedrooms'] < f_bedrooms]
fig = px.histogram(df, x='bedrooms', nbins=19)
c1.plotly_chart(fig, use_container_width=True)

#House per bathrooms
c2.header('Houses per bathrooms')
df = data[data['bathrooms'] < f_bathrooms]
fig = px.histogram(df, x='bathrooms', nbins=19)
c2.plotly_chart(fig, use_container_width=True)

#filters
f_floors = st.sidebar.selectbox('Max number of floor', sorted(set(data['floors'].unique())))
f_waterview = st.sidebar.checkbox('Only Houses With Water View')

c1, c2 = st.beta_columns(2)

#House per floors
c1.header('Houses per floor')
df = data[data['floors'] < f_floors]
fig = px.histogram(df, x='floors', nbins=19)
c1.plotly_chart(fig, use_container_width=True)

#House per whater view
c2.header('Houses With Water Front')

if f_waterview:
    df = data[data['waterfront'] == 1]
else:
    df = data.copy()

fig = px.histogram(df, x='waterfront', nbins=19)
c2.plotly_chart(fig, use_container_width=True)

