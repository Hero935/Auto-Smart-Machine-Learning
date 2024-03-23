# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 21:40:59 2024

@author: user
"""

import os
import datetime
import streamlit as st
import streamlit_antd_components as sac

data_path = r'\\TCTK0FI24\WarRoom$\02_CDTT_PEx\AutoML\data'

def cleanup_logs(logs_directory, max_file_age_days=7):
    try:
        now = datetime.datetime.now()
        max_age = datetime.timedelta(days=max_file_age_days)    
        for filename in os.listdir(logs_directory):
            filepath = os.path.join(logs_directory, filename)
            if os.path.isfile(filepath) and filename.startswith('logs.log'):
                mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(filepath))
                if now - mod_time > max_age:
                    os.remove(filepath)
    except Exception as e:
        st.error(f"An error occurred while cleaning up logs: {e}")

def set_page_configuration():
    st.set_page_config(
        page_title="AutoML",
        page_icon="👋",
        layout="wide",
    )

def show_header():
    st.write("# Auto & Smart Machine Learning! 👋")

def show_features():
    st.subheader('Features', divider='rainbow')
    st.write('Minimize the number of required operations to achieve the following:')
    st.write('- Compare multiple models')
    st.write('- Optimize the model')
    st.write('- Analyze models and generate graphics')
    st.write('- Perform online predictions')
    st.write('- Download the model')

def show_caption():
    st.caption('Website design by :blue[_Scorpio Su_] :sunglasses:')

def show_sidebar():
    st.sidebar.header('Modules', divider='rainbow')
    sidebar_text = """
    Machine Learning use-cases supported on this website:
    
    - **Supervised ML**
        - Regression
        - Classification
    - **Time Series**        
        - Time Series Forecasting
    """
    st.sidebar.markdown(sidebar_text)
    st.sidebar.header('Data Exploration', divider='rainbow')
    sidebar_text = """
    one-line Exploratory Data Analysis (EDA) experience on this website:
    """
    st.sidebar.markdown(sidebar_text)

def run():
    cleanup_logs(os.getcwd())
    set_page_configuration()
    show_header()
    show_features()
    show_caption()
    show_sidebar()

if __name__ == '__main__':
    run()
