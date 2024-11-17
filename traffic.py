# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set up the app title and image
st.title('Traffic Volume Predictor')
st.write("Utilize our advanced machine learning application to predict traffic volume.") 
st.image('traffic_image.gif', use_column_width = True, caption = "Traffic")

# Reading the pickle file that we created before 
xg_pickle = open('XGBoost_traffic.pickle', 'rb') 
xg_model = pickle.load(xg_pickle) 
xg_pickle.close()

# Load the default dataset
default_df = pd.read_csv('Traffic_Volume.csv')
# default_df.dropna()

# Replace NaN values
default_df.fillna({'holiday': 'None', 
                   'temp': default_df['temp'].mean(),
                   'rain_1h': default_df['rain_1h'].mean(), 
                   'snow_1h': default_df['snow_1h'].mean(),
                   'clouds_all': default_df['clouds_all'].mean(), 
                   'weather_main': 'Clear',
                   'traffic_volume': default_df['traffic_volume'].mean()}, 
                   inplace=True)

default_df['date_time'] = pd.to_datetime(default_df['date_time'], format='%m/%d/%y %H:%M')
default_df['hour'] = default_df['date_time'].dt.hour 
default_df['day_of_week'] = default_df['date_time'].dt.dayofweek 
default_df['month'] = default_df['date_time'].dt.month

# Day mapping
day_mapping = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
# Month mapping
month_mapping = {
    1: 'January', 2: 'February', 3: 'March', 4: 'April',
    5: 'May', 6: 'June', 7: 'July', 8: 'August',
    9: 'September', 10: 'October', 11: 'November', 12: 'December'
}

# Map numerical day_of_week and month to their names 
default_df['day_of_week'] = default_df['day_of_week'].map(day_mapping) 
default_df['month'] = default_df['month'].map(month_mapping)

# Drop the date_time column
default_df.drop(columns=['date_time'], axis=1, inplace=True)

##### -------------------   SIDEBAR   ---------------------- ######
# Create a sidebar for input collection
st.sidebar.image('traffic_sidebar.jpg', use_column_width = True, caption = "Traffic volume Predictor")
st.sidebar.header('Input Features')
st.sidebar.write('You can either upload your data file or manually enter input features.')

# Option 1: Asking users to input their data as a file
with st.sidebar.expander('Option 1: Upload CSV file', expanded=False):
    file = st.file_uploader('Upload your CSV file', type=["csv"])
    st.write("Sample Data Format for Upload")
    st.write(default_df.head(5))
    st.warning("Ensure your uploaded file has the same column names and data types as shown above.")
    
# Option 2: Asking users to input their data using a form in the sidebar
with st.sidebar.expander('Option 2: Fill out form', expanded=False):
    st.header("Enter Your Diamond Details")
    
    with st.form('user_inputs_form'):

        # Holiday
        holiday = st.selectbox('Choose whether today is a designated holiday or not.', options=default_df['holiday'].unique()) 

        # Temp
        temp_diff = default_df['temp'].sort_values().diff().dropna() 
        non_zero_temp_diff = temp_diff[temp_diff != 0] 
        temp = st.slider('Average temperature in Kelvin.', min_value=default_df['temp'].min(), 
                                max_value=default_df['temp'].max(), 
                                step=non_zero_temp_diff.min())
        
        # Rain in 1h
        rain_diff = default_df['rain_1h'].sort_values().diff().dropna()
        non_zero_rain_diff = rain_diff[rain_diff != 0]
        rain_1h = st.slider('Amount in mm of rain that occurred in the hour.', min_value=default_df['rain_1h'].min(),
                            max_value=default_df['rain_1h'].max(),
                            step=non_zero_rain_diff.min())
        
        # Snow in 1h
        snow_diff = default_df['snow_1h'].sort_values().diff().dropna()
        non_zero_snow_diff = snow_diff[snow_diff != 0]
        snow_1h = st.slider('Amount in mm of snow that occurred in the hour.', min_value=default_df['snow_1h'].min(),
                            max_value=default_df['snow_1h'].max(),
                            step=non_zero_snow_diff.min())
        
        # Clouds all
        clouds_diff = default_df['clouds_all'].sort_values().diff().dropna()
        non_zero_clouds_diff = clouds_diff[clouds_diff != 0]
        clouds_all = st.slider('Percentage of cloud cover', min_value=default_df['clouds_all'].min(),
                               max_value=default_df['clouds_all'].max(),
                               step=int(non_zero_clouds_diff.min()))

        weather_main = st.selectbox('Choose the current weather', options=default_df['weather_main'].unique()) 

        # month
        month = st.selectbox('Choose month.',  options=default_df['month'].unique()) 

        # day of week
        day_of_week = st.selectbox('Choose day of the week',  options=default_df['day_of_week'].unique()) 

        # hour
        hour = st.selectbox('Choose hour', options=default_df['hour'].unique())

        # Submit Form Button
        submit_button = st.form_submit_button("Submit Form Data")

if file is None:
    # Encode the inputs for model prediction
    encode_df = default_df.copy()
    encode_df = encode_df.drop(columns = ['traffic_volume'])

    # Combine the list of user data as a row to default_df
    encode_df.loc[len(encode_df)] = [holiday,temp,rain_1h,snow_1h,clouds_all,weather_main,hour,day_of_week,month]

    # Create dummies for encode_df
    encode_dummy_df = pd.get_dummies(encode_df)

    # Extract encoded user data
    user_encoded_df = encode_dummy_df.tail(1)

    st.write("")

    # Get the prediction with its intervals
    alpha = st.slider("Select alpha value for prediction intervals", 
                        min_value=0.01, 
                        max_value=0.5, 
                        step=0.01)

    prediction, intervals = xg_model.predict(user_encoded_df, alpha = alpha)
    pred_value = prediction[0]
    lower_limit = intervals[0, :]
    upper_limit = intervals[:, 1][0][0]

    # Ensure limits are within [0, 1]
    lower_limit = max(0, lower_limit[0][0])

    # Show the prediction on the app
    st.write("## Predicting Traffic Volume...")

    # Display results using metric card
    st.metric(label = "Hourly I-94 ATR 301 reported westbound traffic volume", value = f"{pred_value :.0f}")
    st.write(f"With a {(1 - alpha)* 100:.0f}% confidence interval:")
    st.write(f"**Confidence Interval**: [{lower_limit:.2f}, {upper_limit:.2f}]")

else:
    # Loading data
    st.success('**CSV file uploaded successfully!**')
    user_df = pd.read_csv(file) # User provided data
    original_df = pd.read_csv('Traffic_Volume.csv') # Original data to create ML model

    # Replace NaN values
    original_df.fillna({'holiday': 'None', 
                    'temp': original_df['temp'].mean(),
                    'rain_1h': original_df['rain_1h'].mean(), 
                    'snow_1h': original_df['snow_1h'].mean(),
                    'clouds_all': original_df['clouds_all'].mean(), 
                    'weather_main': 'Clear',
                    'traffic_volume': original_df['traffic_volume'].mean()}, 
                    inplace=True)

    original_df['date_time'] = pd.to_datetime(original_df['date_time'], format='%m/%d/%y %H:%M')
    original_df['hour'] = original_df['date_time'].dt.hour 
    original_df['day_of_week'] = original_df['date_time'].dt.dayofweek 
    original_df['month'] = original_df['date_time'].dt.month

    # Day mapping
    day_mapping = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    # Month mapping
    month_mapping = {
        1: 'January', 2: 'February', 3: 'March', 4: 'April',
        5: 'May', 6: 'June', 7: 'July', 8: 'August',
        9: 'September', 10: 'October', 11: 'November', 12: 'December'
    }

    # Map numerical day_of_week and month
    original_df['day_of_week'] = original_df['day_of_week'].map(day_mapping) 
    original_df['month'] = original_df['month'].map(month_mapping)

    # Drop the date_time column
    original_df.drop(columns=['date_time'], axis=1, inplace=True)

    # day_of_week == weekday
    user_df['day_of_week'] = user_df['weekday']

    # Remove output column
    original_df = original_df.drop(columns = ['traffic_volume'])

    # columns order
    user_df = user_df[original_df.columns]
    combined_df = pd.concat([original_df, user_df], axis = 0)

    # Number of rows in original dataframe
    original_rows = original_df.shape[0]

    # Create dummies for the combined dataframe
    combined_df_encoded = pd.get_dummies(combined_df)

    # Split data into original and user dataframes using row index
    original_df_encoded = combined_df_encoded[:original_rows]
    user_encoded_df = combined_df_encoded[original_rows:]

    # Get the prediction with its intervals
    alpha = st.slider("Select alpha value for prediction intervals", 
                        min_value=0.01, 
                        max_value=0.5, 
                        step=0.01)

    prediction, intervals = xg_model.predict(user_encoded_df, alpha = alpha)
    pred_value = prediction[0]

    # Ensure limits are realistic numbers 
    lower_limits = np.maximum(0, intervals[:, 0]) 
    upper_limits = intervals[:, 1] 
    upper_limits = np.maximum(lower_limits, upper_limits)

    # Add the dataframe with the three columns
    # Use original user_df to avoid the dummy variable issue 
    result_df = user_df.copy() 
    result_df['Predicted Volume'] = pred_value 
    result_df['Lower Limit'] = lower_limits 
    result_df['Upper Limit'] = upper_limits

    st.write(f"## Prediction Results with {(1 - alpha)* 100:.0f}% Confidence Interval")
    st.dataframe(result_df)

# Additional tabs for model performance
st.subheader("Model Insights")
tab1, tab2, tab3, tab4 = st.tabs(["Feature Importance", 
                            "Histogram of Residuals", 
                            "Predicted Vs. Actual", 
                            "Coverage Plot"])
with tab1:
    st.write("### Feature Importance")
    st.image('feature_imp.svg')
    st.caption("Relative importance of features in prediction.")
with tab2:
    st.write("### Histogram of Residuals")
    st.image('residual_plot.svg')
    st.caption("Distribution of residuals to evaluate prediction quality.")
with tab3:
    st.write("### Plot of Predicted Vs. Actual")
    st.image('pred_vs_actual.svg')
    st.caption("Visual comparison of predicted and actual values.")
with tab4:
    st.write("### Coverage Plot")
    st.image('coverage.svg')
    st.caption("Range of predictions with confidence intervals.")