# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 21:41:59 2024

@author: user
"""

import streamlit as st
from pycaret.datasets import get_data
import os
root_path = os.getcwd()
#print(f'main: {root_path}')
import pandas as pd
from ydata_profiling import ProfileReport
#from streamlit_pandas_profiling import st_profile_report
import streamlit.components.v1 as components
import datetime

data_path = r'\data'
pkl_path = r'\data\pkl'
st.set_page_config(layout="wide")

@st.cache_data
def create_report(data):
    return ProfileReport(data, title="Profiling Report")

@st.cache_data
def run_upload_data(uploaded_file):
    #file_path = os.path.join(data_path,uploaded_file.name)
    #with open(file_path,"wb") as f:
        #f.write(uploaded_file.getbuffer())
    #data = pd.read_excel(file_path)
    data = pd.read_excel(uploaded_file)
    st.dataframe(data, height=200)
    return data
    
def run():
    st.header('Auto Data Exploration', divider='rainbow')
    data = pd.DataFrame()
    st.sidebar.header('Data Exploration', divider='blue')
    st.sidebar.write("Provide a one-line Exploratory Data Analysis (EDA) experience in a consistent and fast solution")

    st.markdown('<h4 style="color:DeepSkyBlue">1. Dataset preparation</h4>', unsafe_allow_html=True)
    genre = st.radio("Please select a data source", ["Demo dataset", "Upload dataset"], index=None)

    if genre == 'Demo dataset':
        data = get_data('iris')
    elif genre == 'Upload dataset':
        uploaded_file = st.file_uploader("Choose a Excel file", type="xlsx")
        if uploaded_file is not None:
            data = run_upload_data(uploaded_file)

    if not data.empty:
        start_time = datetime.datetime.now()
        st.write('#### 2. Generate report')
        profile = create_report(data)
        #st_profile_report(profile)
        components.html(profile.to_html(), height=550, scrolling=True)
        #st.write('Completed profiling report')
        st.write(f'time taken: {datetime.datetime.now()-start_time}')
        
        st.markdown('<h4 style="color:DeepSkyBlue">3. Download report</h4>', unsafe_allow_html=True)
        html_path = data_path + r"\profiling.html"
        profile.to_file(output_file=html_path)
        with open(html_path, 'rb') as f:
            st.download_button('Download HTML report', f, file_name='profiling.html')

            
if __name__ == '__main__':
    run()
    #data = get_data('iris')
    #data.to_excel('classification_iris.xlsx', index=False)
    #profile = ProfileReport(data, title="Profiling Report")