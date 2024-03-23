# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 21:41:59 2024

@author: user
"""

import streamlit as st
from pycaret.datasets import get_data
from pycaret.time_series import *
import os
root_path = os.getcwd()
#print(f'main: {root_path}')
import datetime
import pandas as pd
#import xgboost
#import catboost

data_path = r'\data'
pkl_path = r'\data\pkl'
st.set_page_config(layout="wide")

@st.cache_data
def run_upload_data(uploaded_file):
    data = pd.read_excel(uploaded_file)    
    st.dataframe(data, height=200)
    return data

@st.cache_data
def run_compare_model(data, option_columns):
# 20W - 5min, 5W - 1min    
    now = datetime.datetime.now()
    if option_columns == '123':
        s = setup(data, session_id = 123)
    else:
        s = setup(data, target=option_columns, session_id = 123)
    expander = st.expander("See Setup Description")
    expander.dataframe(pull())
    st.write('#### 2. Check Stats')    
    st.dataframe(check_stats(), height=200)
    
    st.write('#### 3. Compare Models (sort by MASE)')
    best = compare_models()  # get best model
    compare_table = pull()
    st.dataframe(compare_table, height=200)
    st.write(f'time taken: {datetime.datetime.now()-now}')
    
    expander = st.sidebar.expander("Available models")
    expander.dataframe(models(), height=200)  
    return compare_table

@st.cache_data
def run_tune_predictions(data, option_model):
    now = datetime.datetime.now()
    selected_model = create_model(option_model)   # create model       
    tuned_selected_model = tune_model(selected_model) # tuned model
    expander = st.expander("See performance table")
    expander.dataframe(pull())
    st.write(f'time taken: {datetime.datetime.now()-now}')

    st.write('#### 6. Analyze Model')

    plt1, plt2, plt3, plt4, plt5 = st.tabs(
        ["Forecast", "Forecast-24", "Residuals", 'Diagnostics', 'Insample'
         ])
    with plt1:
        st.write(plot_model(tuned_selected_model, plot = 'forecast', return_fig=True))
    with plt2:
        st.write(plot_model(tuned_selected_model, plot = 'forecast', data_kwargs = {'fh' : 24}, return_fig=True))
    with plt3:
        st.write(plot_model(tuned_selected_model, plot = 'residuals', return_fig=True))
    with plt4:
        st.write(plot_model(tuned_selected_model, plot = 'diagnostics', return_fig=True))
    with plt5:
        st.write(plot_model(tuned_selected_model, plot='insample', return_fig=True))

    st.write('#### 7. Predictions')
    final_selected_model = finalize_model(tuned_selected_model)  # finalize model
    
    st.write('Predictions from data set')
    #predictions = predict_model(final_selected_model, data=data)
    predictions = predict_model(final_selected_model)
    st.dataframe(predictions, height=200)
    return final_selected_model    

def run():
    st.header('Auto Time Series forecasting', divider='rainbow')
    option_columns= ''
    data = pd.DataFrame()
    st.sidebar.header('Time series', divider='blue')
    st.sidebar.write("Time series forecasting is the process of analyzing time series data using statistics and modeling to make predictions and inform strategic decision-making")
    
    st.markdown('<h4 style="color:DeepSkyBlue">1. Dataset preparation</h4>', unsafe_allow_html=True)
    genre = st.radio(
        "Please select a data source",
        ["Demo dataset", "Upload dataset"],
        index=None,
    )
    if genre == 'Demo dataset':
        data = get_data('airline')
        st.dataframe(data, height=200)        
        st.pyplot(data.plot().figure)
        option_columns = '123'
    elif genre == 'Upload dataset':
        uploaded_file = st.file_uploader("Choose a Excel file", key="file_uploader_1", type="xlsx")
        if uploaded_file is not None:
            data = run_upload_data(uploaded_file)      
            st.markdown('<h4 style="color:DeepSkyBlue">2. Set Target</h4>', unsafe_allow_html=True)
            option_columns = st.selectbox(
                'Which column is set as the target?',
                list(data.columns), index=None)
    if option_columns != '' and option_columns is not None:
        compare_table = run_compare_model(data, option_columns)
        
        #st.write('#### 4. Select best Model')
        st.markdown('<h4 style="color:DodgerBlue">4. Select Model</h4>', unsafe_allow_html=True)
        option_model = st.selectbox(
            'Which model to choose for analysis?',
            compare_table.index.tolist())
        if option_model is not None:
            st.write('#### 5. Tune model')
            final_selected_model = run_tune_predictions(data, option_model)                
            #st.write('#### 6. Analyze Model')                   
            #st.write('#### 7. Predictions')
    
            st.markdown('<h4 style="color:DeepSkyBlue">8. Online Prediction</h4>', unsafe_allow_html=True)
            #st.write('Predictions from input')
            #new_data = pd.DataFrame(columns=data.columns)
            #new_data = new_data.append(data.iloc[0], ignore_index=True)
            #new_data = new_data.drop(columns=[option_columns])
            #edited_df = st.data_editor(new_data.transpose())
            #output = ''
            #if st.button('Prediction'):                
            #    output = predict_model(final_selected_model, data=edited_df.transpose())['prediction_label'][0]
            #    st.success(f'👉 *{option_columns}* is *{format(output)}*')
                
            st.write('- Predictions from new data')
            prediction_file = st.file_uploader("Choose a Excel file", key="file_uploader_2", type="xlsx")
            if prediction_file is not None:
                data = run_upload_data(prediction_file)   
                predicted_results = predict_model(final_selected_model, data=data)
                st.write('- Results')
                st.dataframe(predicted_results, height=200)
            
            #st.write('#### 9. Save model')
            filename = 'my_Time_Series_model'
            st.markdown('<h4 style="color:DeepSkyBlue">9. Save model</h4>', unsafe_allow_html=True)
            file_path = os.path.join(pkl_path, filename)
            save_model(final_selected_model, file_path)
            with open(file_path + '.pkl', "rb") as file:
                st.download_button(
                    label="Download final model",
                    data=file,
                    file_name=os.path.basename(file_path),
                )
            
if __name__ == '__main__':
    run()
    #data = get_data('iris')
    #data.to_excel('classification_iris.xlsx', index=False)