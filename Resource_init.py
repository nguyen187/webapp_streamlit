import streamlit as st
import pandas as pd
import numpy as np
import datetime
import pyodbc
import xgboost as xgb
import joblib
import time
from keras.models import load_model
from scipy import signal
from scipy.signal import savgol_filter
import seaborn as sns
import matplotlib.pyplot as plt
from markdown import markdown
from PIL import Image
# Khởi tạo biến toàn cục connection pool
# pool = None

def get_connection_string():
    # Trả về chuỗi kết nối database của bạn
    server = st.secrets["server"]
    database = st.secrets["database"]
    user = st.secrets["username"]
    passwd = st.secrets["password"]
    return f"driver={{ODBC Driver 17 for SQL Server}}; server={server}; database={database};UID={user}; PWD={passwd}"

# def get_pool():
#     # Tạo pool kết nối nếu chưa được tạo
#     global pool
#     if pool is None:
#         connection_string = get_connection_string()
#         pool = ConnectionPool(minimum=1, maximum=10, connection_string=connection_string)
#     return pool

# @st.cache(hash_funcs={pyodbc.Connection: id})
@st.cache_resource
def get_connection():
    # Lấy một kết nối từ pool
    connection_string = get_connection_string()
    conx = pyodbc.connect(connection_string)
    return conx

def get_cursor():
    # Tạo con trỏ để truy vấn database
    connection = get_connection()
    cursor = connection.cursor()
    return connection,cursor

def close_connection(conx):
    return conx.close()
    

# Khởi tạo biến toàn cục model
xgb_model_raman = None
scale = None
ann_model_process = None
@st.cache_resource
def load_model_resource():
    # Đọc model nếu chưa được đọc
    global xgb_model_raman,scale,ann_model_process
    
    if xgb_model_raman is None:
        xgb_model_raman = xgb.XGBRegressor()
        xgb_model_raman.load_model("model_xgb.json")
        scale = joblib.load('scale.pkl')
        ann_model_process = load_model("network.h5")
    return xgb_model_raman,scale,ann_model_process


USERS = {
            'test@localhost.com',
            'thanhnguyen187201@gmail.com',
            'trinhntrung@gmail.com'
        }
server = st.secrets["server"]
database = st.secrets["database"]
user = st.secrets["username"]
passwd = st.secrets["password"]
@st.cache_resource
def Login(w =False):
    if st.experimental_user.email in USERS:
        if w == True:
            st.write("# Welcome to BiaPharm Website! 👋")
            st.write('Hello, %s!' % st.experimental_user.email)
            st.write('server: {sv}'.format(sv = server))
            image = Image.open('anhbia.png')
            st.image(image)
        return True
    else:
        st.error("Your account don't permission", icon="🚨")
        return False
