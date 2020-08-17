# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest CLassifier model

MODEL_PATH = 'C:\\Users\\eponr\\Desktop\\Open_Source_project\\2. Bike Sharing dataset\\model'
MODEL_NAME = 'XGB_linear_model_with_r2Value_0.77.pkl'


regressor = pickle.load(open(MODEL_PATH + '/' +  MODEL_NAME, 'rb'))




app = Flask(__name__)


@app.route('/')
def home():
	return render_template('index.html')



@app.route('/predict', methods=['POST'])
def predict():
    temp_array = list()
    
    if request.method == 'POST':
        
        season = request.form['Which Season ?']
        if season == 'Spring':
            temp_array = temp_array + [0]
        elif season == 'Summer':
            temp_array = temp_array + [1]
        elif season == 'Fall':
            temp_array = temp_array + [2]
        elif season == 'Winter':
            temp_array = temp_array + [3]
            
            
        year = int(request.form['Which Year ?'])
        if year == 2011:
            temp_array = temp_array + [0]
        elif year == 2012:
            temp_array = temp_array + [1]
            
        
        
        month = request.form['Which Month ?']
        
        if month == 'January':
            temp_array = temp_array + [0]
        elif month == 'February':
            temp_array = temp_array + [1]
        elif month == 'March':
            temp_array = temp_array + [2]
        elif month == 'April':
            temp_array = temp_array + [3]
        elif month == 'May':
            temp_array = temp_array + [4]
        elif month == 'June':
            temp_array = temp_array + [5]
        elif month == 'July':
            temp_array = temp_array + [6]
        elif month == 'August':
            temp_array = temp_array + [7]
        elif month == 'September':
            temp_array = temp_array + [8]
        elif month == 'October':
            temp_array = temp_array + [9]
        elif month == 'November':
            temp_array = temp_array + [10]
        elif month == 'December':
            temp_array = temp_array + [11]
        
        
        holiday = request.form['Holiday or Not ?']
        if holiday == 'Yes':
            temp_array = temp_array + [0]
        elif holiday == 'No':
            temp_array = temp_array + [1]
        
        
        day = request.form['Which day ?']
        if day == 'Sunday':
            temp_array = temp_array + [0]
        elif day == 'Monday':
            temp_array = temp_array + [1]
        elif day == 'Tuesday':
            temp_array = temp_array + [2]
        elif day == 'Wednesday':
            temp_array = temp_array + [3]
        elif day == 'Thursday':
            temp_array = temp_array + [4]
        elif day == 'Friday':
            temp_array = temp_array + [5]
        elif day == 'Saturday':
            temp_array = temp_array + [6]
        
        
        workingday = request.form['Working day or Not ?']
        if workingday == 'Yes':
            temp_array = temp_array + [0]
        elif workingday == 'No':
            temp_array = temp_array + [1]
        
        
        weathersit = request.form['Which Weather condition ?']
        if weathersit == 'Clear & Cloudy':
            temp_array = temp_array + [0]
        elif weathersit == 'Mist & Few clouds':
            temp_array = temp_array + [1]
        elif weathersit == 'Rain & Thundderstorm':
            temp_array = temp_array + [2]
        elif weathersit == 'Heavy Rain & Thunderstorm':
            temp_array = temp_array + [3]
            
        atemp = float(request.form['atemp'])/50
        humid = float(request.form['hum'])/100
        windspeed = float(request.form['windspeed'])/67
        
    
        temp_array = temp_array + [atemp, humid, windspeed]
        
        data = np.array([temp_array])
        print('data :', data, data.shape)
        my_prediction = int(regressor.predict(data)[0])
        print(my_prediction)
              
        return render_template('result.html', predicted_usage_cnt = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)