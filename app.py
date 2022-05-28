import streamlit as st
from SVM import *
from SVM import show_predict_page
# from predict_page import select_country
# from predict_page import model
from SVM import display
from SVM import *
from explore_page import top3
from explore_page import fig 
# from explore_page import fig2
from explore_page import image
from explore_page import goos
from explore_page import foos
# from explore_page import imagep
from PIL import Image
import time



with st.spinner(text='In progress'):
    time.sleep(1.5)
    st.success('Done')
st.sidebar.title('Explore Or Predict')
imagep = Image.open('survey.jpg')
page = st.sidebar.selectbox("", ("Predict", "Explore"))
show = st.sidebar.checkbox("Your Dataset")
st.sidebar.image(imagep,caption='Designed by dugu', width=305)
st.sidebar.write(" ")
st.sidebar.write(" ")
st.sidebar.write(" ")
st.sidebar.write(" ")
st.sidebar.write(" ")
st.sidebar.write(" ")
st.sidebar.write(" ")
st.sidebar.write("For more information about COVID 19 [google.com](https://www.google.com/search?q=covid+19+statistics&rlz=1C1SQJL_en__904__906&oq=cov&aqs=chrome.2.69i60j69i57j69i59l2j69i60l3j69i65.3515j0j7&sourceid=chrome&ie=UTF-8#colocmid=/m/02j71&coasync=0)")



if page == "Predict":
    if show:
        st.write("Your dataset sorted by no. of confirmed cases (Last updated !)")
        display
    # File.tail(100)
    # st.checkbox("Your Dataset")
    show_predict_page()
    # st.image(imagep,caption='Designed by dugu', width=300)
    # if model == "Linear Regression": 
    # print(c)
    # print(c) 
#     if model == "Exponential Smoothing" and country == "India":
#         ES = st.button("Predict")
#     if ES:
#         st.subheader(f"{fcast1[0]:.2f}")
elif page == "Explore":
    st.image(image,caption='Information', width=250)
    st.title("Covid 19 Statistics")
    st.subheader("Total Cases:")
    st.subheader(goos)
    st.subheader("Total Deaths:")
    # st.json({'Cases': goos,'Deaths': foos})
    st.subheader(foos)
    top3
    st.plotly_chart(fig)


    st.write("Reference from [Wikipedia.org](https://en.wikipedia.org/wiki/Template:COVID-19_pandemic_data)")
    # st.write("For more information about COVID 19 [google.com](https://www.google.com/search?q=covid+19+statistics&rlz=1C1SQJL_en__904__906&oq=cov&aqs=chrome.2.69i60j69i57j69i59l2j69i60l3j69i65.3515j0j7&sourceid=chrome&ie=UTF-8#colocmid=/m/02j71&coasync=0)")

    # Click here for more information about updated numbers on the COVID-19 pandemic's confirmed cases and deaths

    # st.plotly_chart(fig2)

# elif page == "Map":
#     import plotly.express as px
#     fig = px.scatter_geo(display, locations="Country",
#                          size="Confirmed", # size of markers, "pop" is one of the columns of gapminder
#                          )
#     fig.show()
# page = st.sidebar.selectbox("", ("Predict", "Explore", "Map"))
 