import streamlit as st
import pandas as pd
import matplotlib as plt
from SVM import File

import plotly.express as px
from PIL import Image
import numpy as np
import seaborn as sns
 
url = 'https://en.wikipedia.org/wiki/Template:COVID-19_pandemic_data'
data = pd.read_html(url)[0]
see = data.drop(['Unnamed: 0'], axis = 1)
# see.to_csv('covid 19 statistics.csv')
 

see = see[['Location','Cases','Deaths']]
#reaming column names
see.columns = ['Location','Cases','Deaths']
goos = see["Cases"].values[0]
foos = see["Deaths"].values[0]

# country = File[File.Country == 'India']
# desce=country.sort_values(by="Confirmed",ascending=False)
# top = desce.head()

# teritory=File.sort_values(by="Confirmed",ascending=False)
# top = teritory["Country"].unique()
# top1 = teritory["Confirmed"].unique()
# top2 = pd.concat([top, top1], ignore_index=True)

# top = df.sort_values(['ID2', 'counter'], ascending=[True, False]).drop_duplicates(['ID2']).reset_index(drop=True)

see.sort_index()#see.sort_values('Cases')
# see.iloc[1: , :]
see.drop(see.head(1).index,inplace=True) # drop first n rows                           #TO REMOVE FIRST ROW OF DATAFRAME
see.drop(see.tail(1).index,inplace=True) # drop last n rows
# see['Deaths'].dropna()
# see['Cases'].replace('4', ' ', regex=True)
# see['Deaths'] = see['Deaths'].replace(['—'],'0')    #  111111111111111111111111111111111111111111111111111111111111111111111  replace all in dataframe
top3 = see
top = top3.head(10)
top['Deaths'] = top['Deaths'].astype('float')



fig = px.pie(top, values='Cases', names='Location',
             title='Top 10 countries cases percentage',
             hover_data=['Deaths'], labels={'Country':'Cases'})
fig.update_traces(textposition='inside', textinfo='percent+label')
# st.plotly_chart(fig)

# fig2 = px.bar(x='Location', y='Deaths')

# fig2 = sns.set(rc={'figure.figsize':(15,10)}) 
# sns.barplot(x="Location",y="Deaths",data=top,hue="Location")
# plt.show()


image = Image.open('covid.jpg')
# imagep = Image.open('D:\SEMI FINAL\covidpredict.png')
# total = world["Cases"].values[0]


# data = File["Country"].value_counts()

# fig, ax1 = plt.subplot() 
# ax1.pie(data, labels=data.index, autopct="%1.1f%%", shodow=True, startangle=90)
# ax1.axis("equal")

# plt.pie(data,labels=my_labels,autopct='%1.1f%%')
# plt.title('My Title')
# plt.axis('equal')
# plt.show()

# top = top3.groupby(['Country']).sum().plot(kind='pie', y='Confirmed')
# top = top3.groupby(['Country']).sum().plot(kind='pie', y='Confirmed', autopct='%1.0f%%',title='Points Scored by Team')











# maxValues = desce.max()
 
# print(maxValues)
    # # print(country)
    # #keeping only required colomns
    # country = country[['Confirmed']]
    # #renaming column names
    # country.columns = ['Confirmed']
#     return File
# File["Country"].unique()
# st.dataframe(country)
# File = load_data()

# def show_explore_page():
#     st.title("Explore Cases of Different Countries")
#     st.write(
#         """ ### Covid cases in different Countries """
    # )

    # data = 