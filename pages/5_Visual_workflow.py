# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 21:41:59 2024

@author: user
"""

#import os
import pandas as pd
import streamlit.components.v1 as components
import streamlit as st
import pygwalker as pyg
#from pycaret.datasets import get_data
import datetime

data_path = r'\data'
pkl_path = r'\data\pkl'
st.set_page_config(layout="wide")

@st.cache_data
def run_upload_data(uploaded_file):
    #file_path = os.path.join(data_path,uploaded_file.name)
    #with open(file_path,"wb") as f:
        #f.write(uploaded_file.getbuffer())
    #data = pd.read_excel(file_path)
    data = pd.read_excel(uploaded_file)
    st.dataframe(data, height=200)
    return data

@st.cache_resource
def get_pyg_html(df: pd.DataFrame) -> str:
    #html = get_streamlit_html(df, spec="./gw0.json", use_kernel_calc=True, debug=False)
    pyg_html = pyg.to_html(df)
    return pyg_html

    
def run():
    st.header('Visual workflow', divider='rainbow')
    st.sidebar.header('Visual workflow', divider='blue')
    st.sidebar.write("Exploratory Data Analysis with Visual workflow")
    
    st.markdown('<h4 style="color:DeepSkyBlue">1. Dataset preparation</h4>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a Excel file", key="file_uploader_1", type="xlsx")
    if uploaded_file is not None:
        data = run_upload_data(uploaded_file)
        #init_streamlit_comm()
        st.markdown('<h4 style="color:DeepSkyBlue">2. Visual workflow</h4>', unsafe_allow_html=True)
        start_time = datetime.datetime.now()
        components.html(get_pyg_html(data), height=1000, scrolling=True)
        st.write(f'time taken: {datetime.datetime.now()-start_time}')
            
if __name__ == '__main__':
    run()
    #data = get_data('iris')
    #data.to_excel('classification_iris.xlsx', index=False)
    #profile = ProfileReport(data, title="Profiling Report")