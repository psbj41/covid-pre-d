from PIL import Image

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import timedelta
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from statsmodels.tsa.api import Holt
from sklearn.metrics import r2_score


import streamlit as st  
import pickle 
import numpy as np 


imagep = Image.open('covidpredict.png')
# st.image(imagep, width=160)
st.image(imagep, width=200, use_column_width=False, clamp=True, channels="RGB", output_format="auto")
# st.write('Upload at least one file !')

Files = st.file_uploader("Choose a file")
# if Files is not None:
# # if not (Files is None):
#     File = pd.read_csv(Files)

if not Files:
    st.warning("Please upload a File before proceeding!")
    st.stop()  
else:
    File = pd.read_csv(Files) 


 
    # st.write('## Data set')
    # st.dataframe(df,3000,500)
#keeping only required colomns 
File = File[['Date','Country','Confirmed','Deaths']]
#renaming column names
File.columns = ['Date','Country','Confirmed','Deaths'] 

# display = File.tail(100)
display = File.sort_values(['Confirmed'], ascending=[False]).drop_duplicates(['Country']).reset_index(drop=True)

# imagep = Image.open('D:\SEMI FINAL\covidpredict.png')
# st.image(imagep,caption='Designed by dugu', width=160)
# df.columns[1]

def show_predict_page():
    # imagep = Image.open('D:\SEMI FINAL\covidpredict.png')
    # st.image(imagep, width=160)
    st.title("Covid 19 Estimate")

    # st.write("We need some information to predict the Cases")
    
    # countries = {
    #     "India",
    #     "United States",
    #     "United Kingdom",
    #     "Germany",
    #     "Brazil",
    #     "France",
    #     "Spain",
    #     "Australia",
    #     "Italy"
    # }

    models = {
        "Linear Regression",
        "Support Vector Machine",
        "Time Series Analysis",
        "Exponential Smoothing"
    }
    # data_series = data['Country'].tolist()
    data_series = File["Country"].unique()
    s = st.selectbox("Country", data_series)

    model = st.selectbox("Models", models)
    # select_country(s)
    if model == "Support Vector Machine":
        SVM = st.button("Predict")
        # if not SVM:
        #     st.write("Choose a Country and Model for predict !")
        if SVM:
            country = File[File.Country == s]
            c = country.sort_values('Confirmed', ascending=True).drop_duplicates(['Confirmed'])
            country["Date"] = pd.to_datetime(country["Date"])
            datewise = country.groupby(["Date"]).agg({"Confirmed":"sum","Deaths":"sum"})
            datewise["Days Since"]=datewise.index-datewise.index[0]
            datewise["Days Since"]=datewise["Days Since"].dt.days
            train_ml = datewise.iloc[:int(datewise.shape[0]*0.95)]
            valid_ml = datewise.iloc[:int(datewise.shape[0]*0.95)]
            model_scores=[]
            svm = SVR(C=1,degree=5,kernel='poly',epsilon=0.001)
            svm.fit(np.array(train_ml["Days Since"]).reshape(-1,1),np.array(train_ml["Confirmed"]).reshape(-1,1))
            prediction_valid_svm = svm.predict(np.array(valid_ml["Days Since"]).reshape(-1,1))
            new_date = []
            new_prediction_svm = []
            for i in range(1,30):
                new_date.append(datewise.index[-1]+timedelta(days=i))
                new_prediction_svm.append(svm.predict(np.array(datewise["Days Since"].max()+i).reshape(-1,1))[0])
            pd.set_option("display.float_format",lambda x: '%.f'%x)
            model_predictions=pd.DataFrame(zip(new_date,new_prediction_svm),columns = ["Dates","Support Vector Machine"])
            st.table(model_predictions.head(5))       
            
            d = plt.figure(figsize=(10,6))
            plt.plot(country['Date'],country['Confirmed']) 
            plt.plot(new_date,new_prediction_svm, color = 'r')
            # plt.plot(fit1.forecast(30))
            plt.xlabel("Dates")
            plt.ylabel("Number of Cases")
            plt.title(f"Cases in {s}")
            plt.grid()
            st.pyplot(d)     
                                                                                                


    elif model == "Linear Regression":
        LR = st.button("Predict")
        # if not LR:
        #     st.write("Choose a Country and Model for predict !")
        if LR:
            # pr = st.subheader(f"{s}")
            # sel = select_country(s)
            country = File[File.Country == s]
            c = country.sort_values('Confirmed', ascending=True).drop_duplicates(['Confirmed'])
            # print(c.tail(2))
            # st.subheader(f"{country}")

            country["Date"] = pd.to_datetime(country["Date"])

            datewise = country.groupby(["Date"]).agg({"Confirmed":"sum","Deaths":"sum"})

            datewise["Days Since"]=datewise.index-datewise.index[0]
            datewise["Days Since"]=datewise["Days Since"].dt.days
            train_ml = datewise.iloc[:int(datewise.shape[0]*0.95)]
            valid_ml = datewise.iloc[:int(datewise.shape[0]*0.95)]
            model_scores=[]

            l = LinearRegression(normalize=True)
            l.fit(np.array(train_ml["Days Since"]).reshape(-1,1),np.array(train_ml["Confirmed"]).reshape(-1,1))

            prediction_valid_lin_reg = l.predict(np.array(valid_ml["Days Since"]).reshape(-1,1))

            new_date = []
            new_prediction_lr = []
            for i in range(1,30):
                new_date.append(datewise.index[-1]+timedelta(days=i))
                new_prediction_lr.append(l.predict(np.array(datewise["Days Since"].max()+i).reshape(-1,1))[0][0])
            pd.set_option("display.float_format",lambda x: '%.f'%x)
            model_predictions=pd.DataFrame(zip(new_date,new_prediction_lr),columns = ["Dates","Linear Regression"])

            # print(model_predictions)
            # st.subheader(f"{model_predictions}")                                            # worked but not that good
            # st.dataframe(model_predictions)                                                 #   Worked but small
            st.table(model_predictions.head(5))               
            #st.write("The R^2 Score for model is :",r2score)
            d = plt.figure(figsize=(10,6))
            plt.plot(country['Date'],country['Confirmed'])
            plt.plot(new_date,new_prediction_lr, color = 'r')
            # plt.plot(fit1.forecast(30))
            plt.xlabel("Dates")
            plt.ylabel("Number of Cases")
            plt.title(f"Cases in {s}")
            plt.grid()
            st.pyplot(d)                                                                             # GOOOD WORKED


    elif model == "Time Series Analysis":
        TSA = st.button("Predict")
        # if not TSA:
        #     st.write("Choose a Country and Model for predict !")
        if TSA:
            # pr = st.subheader(f"{s}")
            # sel = select_country(s)
            country = File[File.Country == s]
            c = country.sort_values('Confirmed', ascending=True).drop_duplicates(['Confirmed'])
            # print(c.tail(2))
            # st.subheader(f"{country}")

            country["Date"] = pd.to_datetime(country["Date"])

            datewise = country.groupby(["Date"]).agg({"Confirmed":"sum","Deaths":"sum"})

            datewise["Days Since"]=datewise.index-datewise.index[0]
            datewise["Days Since"]=datewise["Days Since"].dt.days
            model_train=datewise.iloc[:int(datewise.shape[0]*0.85)]
            valid=datewise.iloc[int(datewise.shape[0]*0.85):]


            holt = Holt(np.asarray(model_train["Confirmed"])).fit(smoothing_level=1.4,smoothing_slope=0.2)
            y_pred = valid.copy()
            y_pred["Holt"]=holt.forecast(len(valid))

            holt_new_data=[]
            holt_new_prediction=[]
            for i in range(1,30):
                holt_new_data.append(datewise.index[-10]+timedelta(days=i))
                holt_new_prediction.append(holt.forecast((len(valid)+i))[-1])
            # model_predictions["Holts Linear Model Prediction TSA"]=holt_new_prediction
            model_predictions=pd.DataFrame(zip(holt_new_data,holt_new_prediction),columns = ["Dates","TSA"])
            ron =pd.DataFrame(zip(holt_new_data,holt_new_prediction),columns = ["Date","Confirmed"])

            # print(model_predictions)
            st.table(model_predictions.head(5))  

            d = plt.figure(figsize=(10,6))
            plt.plot(country['Date'],country['Confirmed'])
            plt.plot(holt_new_data,holt_new_prediction, color = 'r')
            # plt.plot(fit1.forecast(30))
            plt.xlabel("Dates")
            plt.ylabel("Number of Cases")
            plt.title(f"Cases in {s}")
            plt.grid()
            st.pyplot(d)

            # wide_df = px.data.medals_wide()
            # kang = country[["Date","Confirmed"]]
            # kong = ron.loc[ron['Confirmed'] >= 39257080]                                       #df.loc[df[‘Price’] >= 10]
    

            # fig = px.bar(wide_df, x="nation", y=["gold", "silver", "bronze"], title="Wide-Form Input, relabelled",
            #             labels={"value": "count", "variable": "medal"})
            # fig.show()

            # import plotly.express as px
            # wide_df = px.data.medals_wide(model_predictions)
            # st.plotly_chart(wide_df) 

            # begin_date = '2022-01-24'

            # df = pd.DataFrame({'Confirmed':holt_new_prediction,
            #                    'Date':pd.date_range(begin_date, periods=len(Confirmed))})
            # print (df.head(10))

            # import plotly.express as px
            # # fige = px.line(data, color_discrete_map={s: 'red'}) 
            # con = country[["Date","Confirmed"]]
            # kon = ron[["Date","Confirmed"]]      #2022-01-23
            # print(con)
            # print(kon)
            # fige = px.line(con.append(kon))
            # st.plotly_chart(fige) 

    elif model == "Exponential Smoothing":
        ES = st.button("Predict")
        # if not ES:
        #     st.write("Choose a Country and Model for predict !")
        if ES:
            country = File[File.Country == s]
            country = country[['Confirmed']]
            country.columns = ['Confirmed']
            
            # df[['A','B']] = df[['A','B']].apply(pd.to_datetime) #if conversion required
            # df['C'] = (df['B'] - df['A']).dt.days
            
            from statsmodels.tsa.api import SimpleExpSmoothing
            index= pd.date_range('22/1/2020', periods=len(country))   #
            confi = country['Confirmed'].to_list()
            data = pd.Series(confi, index)

            fit1 = Holt(data).fit(smoothing_level = 0.8, smoothing_slope = 0.2, optimized = False)
            fit1.fittedvalues
            fcast1 = fit1.forecast(5). rename("Exponential Smoothing")
            # st.checkbox("Your Dataset")
            # st.subheader(f"{fcast1[0]:.2f}")
            st.table(fcast1)
            # data.to_dict()
            # print(data)
            d = plt.figure(figsize=(10,6))
            plt.plot(index,confi)
            plt.plot(fit1.forecast(30), color = 'r')
            plt.xlabel("Dates")
            plt.ylabel("Number of Cases")
            plt.title(f"Cases in {s}")
            plt.grid()
            st.pyplot(d)


            # while d:







            # data.plot(figsize=(12,8)) 
            # fit1.forecast(15).plot()
            # st.line_chart(data.plot(figsize=(12,8)))
            # while '2022-01-23' in index != True:
            #     p = color_discrete_map={s : 'blue'}
            # else:
            #     q = color_discrete_map={s : 'red'}
            

            # st.plotly_chart(d)
            # import plotly.express as px #88888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888
            # # fige = px.line(data, color_discrete_map={s: 'red'}) 
            # fige = px.line(data.append(fit1.forecast(30)))#88888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888
            # st.plotly_chart(fige) #88888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888
    
            # # confi.drop(confi.head(0).index,inplace=True) # drop first n rows  
            # confi.pop(0)
            # df_train = confi[:-24]
            # df_test = confi[-24:]

            # # Plot
            # plt.title('Airline passengers train and test sets', size=20)
            # plt.plot(df_train['Confirmed'], label='Training data')
            # plt.plot(df_test['Confirmed'], color='gray', label='Testing data')
            # plt.legend();
			
			
			