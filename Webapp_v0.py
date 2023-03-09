from markdown import markdown
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import pyodbc
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import xgboost as xgb
import joblib
import time
import tensorflow as tf
from keras.models import load_model
from scipy import signal
from scipy.signal import savgol_filter
# Show extra buttons for admin users.
USERS = {
    'test@localhost.com',
    'thanhnguyen187201@gmail.com',
    'trinhntrung@gmail.com',
}
server = st.secrets["server"]
database = st.secrets["database"]
user = st.secrets["username"]
passwd = st.secrets["password"]
if st.experimental_user.email in USERS:
    st.write('Hello, %s!' % st.experimental_user.email)
    st.write('server:',st.secrets["server"])
    st.write('database:',st.secrets["database"])
    st.write('username:',st.secrets["username"])
  
    
    @st.cache_resource
    def load_model_predict():
        xgb_model_raman = xgb.XGBRegressor()
        xgb_model_raman.load_model("model_xgb.json")
        scale = joblib.load('scale.pkl')
        ann_model_process = load_model("network.h5")
        return xgb_model_raman,scale,ann_model_process
    xgb_model_raman,scale,ann_model_process = load_model_predict()
    @st.cache_resource
    def db_connection():
        #-----------initial connect to sql server------------------
        # conx = pyodbc.connect("driver={SQL Server}; server=20.5.100.16; database=BioPharm;UID=sa; PWD=nguyen187201@Abc")#ket noi database
        # conx = pyodbc.connect("Driver={ODBC Driver 17 for SQL Server};Server=10.14.171.59;Database=BioPharm;UID=sa;PWD=nguyen187201@Abc")
        conx = pyodbc.connect("Driver={ODBC Driver 17 for SQL Server};Server="+server +";Database=" +database+";UID=" +user +";PWD="+ passwd ) 

        cursor = conx.cursor()# khoi tao ket noi
        return conx,cursor
    conx, cursor = db_connection()

    # def init_connection():
    #     return pyodbc.connect(
    #         "DRIVER={SQL Server};SERVER="
    #         + st.secrets["server"]
    #         + ";DATABASE="
    #         + st.secrets["database"]
    #         + ";UID="
    #         + st.secrets["username"]
    #         + ";PWD="
    #         + st.secrets["password"]
    #     )

    # conx = init_connection()
    # cursor = conx.cursor()
    def predict(input_df1,input_df2):
        
        X_pro = input_df1
        X_pro['Sugar mass flow'] = X_pro['Sugar feed rate(Fs:L/h)']*X_pro['Substrate concentration(S:g/L)']
        Col1=['Sugar mass flow','Water for injection/dilution(Fw:L/h)','Temperature(T:K)','Dissolved oxygen concentration(DO2:mg/L)','Vessel Volume(V:L)','pH(pH:pH)','Temperature(T:K)']
        X_pro = X_pro[Col1]
            #scale = StandardScaler()
        X_pro = scale.fit_transform(X_pro)
        y_head_pro = ann_model_process.predict(X_pro)
        input_df1 = input_df1.drop(['Sugar mass flow'],axis = 1)
        #for raman

        input_df2 = input_df2.drop(['202','201'],axis=1)
        input_df2 = input_df2.iloc[:,1100:1350]

        #smooth signal
        X_ra1 = signal.savgol_filter(input_df2.values, window_length=11, polyorder=3, mode="nearest")
        X_ra1 = savgol_filter(X_ra1, 17, polyorder = 3,deriv=2)
        y_head_ra = xgb_model_raman.predict(X_ra1)
        # create a array full cus_ID with row = row data input
        df_cus_ID = np.full((input_df1.shape[0], 1), cus_ID)
        df_pro_ID = np.full((input_df1.shape[0], 1), pro_ID)
        df_batch_ID = np.full((input_df1.shape[0], 1), batch_ID)
        
        input_df1['Penicillin concentration(P:g/L)'] = y_head_ra
        input_df1['predict_Pen'] = y_head_pro
        input_df1 = pd.concat([pd.DataFrame(df_cus_ID, columns=['Cust']), pd.DataFrame(df_pro_ID, columns=['Project_ID']), pd.DataFrame(df_batch_ID, columns=['2-PAT control(PAT_ref:PAT ref)']),
                            input_df1,input_df2], axis=1)  # include cus_id,pro_id,test_id,data
        result_df = load_data(input_df1)
        st.header('Result')
    

        progress_text = "Operation in progress. Please wait."
        my_bar = st.progress(0, text=progress_text)

        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1, text=progress_text)
        st.dataframe(result_df)
        result_df = result_df.fillna(value=0)
        return result_df
    
    def insert_sql(data):
        # self.df.rename(columns={"2-PAT control(PAT_ref:PAT ref)": "Batch ID", "Batch ID": "2-PAT control(PAT_ref:PAT ref)"})
        sql_columns = ','.join(['"{}"'.format(col) for col in data.columns])
        sql_values = ','.join(['?'] * len(data.columns))
        insert_exp_sql = '''
        INSERT INTO {table} ({columns})
        VALUES ({values})
        '''.format(table='db_dboard', columns=sql_columns, values=sql_values)
        for row in data.itertuples():
            cursor.execute(insert_exp_sql, row[1:])
        conx.commit()
        st.success('This is a success message!', icon="✅")
        st.balloons()
    # self.Display_data(self.df)
    st.write("""
    #Applied Energy Prediction App
    """ )
    st.sidebar.header('Input Feature')
    st.sidebar.markdown("""
    Example input
    """)
    
    input_df1 = pd.DataFrame()
    input_df2 = pd.DataFrame()
    result_df = pd.DataFrame()
    
    #Collect user input feature into dataframe
    upload_file1 = st.sidebar.file_uploader('Upload your input csv file process',type=['csv'])
    upload_file2 = st.sidebar.file_uploader('Upload your input csv file raman',type=['csv'])

    if upload_file1 is not None and upload_file2 is not None:
        input_df1 = pd.read_csv(upload_file1)
        input_df2 = pd.read_csv(upload_file2)
        # Cache the data frame so it's only loaded once
        @st.cache_data
        def load_data(input_df):
            return input_df
        # Boolean to resize the dataframe, stored as a session state variable
        st.checkbox("Use container width", value=False, key="use_container_width")
        df1= load_data(input_df1)
        df2= load_data(input_df2)
        # Display the dataframe and allow the user to stretch the dataframe
        # across the full width of the container, based on the checkbox value
        st.header('Data 1:')
        st.dataframe(df1, use_container_width=st.session_state.use_container_width)
        st.header('Data 2:')
        st.dataframe(df2, use_container_width=st.session_state.use_container_width)
        
        cus_ID = st.sidebar.text_input('Cus ID:')
        pro_ID = st.sidebar.text_input('Project ID:')
        batch_ID = st.sidebar.text_input('Batch ID (only numeric):')
        experimental_date  = st.sidebar.date_input(
            "Experimental date is",
            datetime.date(2019, 7, 6))
        st.sidebar.write('Experimental date is:', experimental_date )
        submit_time = datetime.datetime.now().strftime(
                    "%m/%d/%Y %H:%M:%S")  # save time now
        if st.sidebar.button('Submit'):
            result_df = predict(input_df1,input_df2)
            insert_sql(result_df)
        
else:
    st.error('This is an error', icon="🚨")
    


    