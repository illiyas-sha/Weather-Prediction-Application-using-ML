from flask import Flask,render_template,url_for,request,jsonify
from flask_cors import cross_origin
import datetime
import pickle
import pandas as pd
import joblib

app = Flask(__name__, template_folder="template")

model_fit = joblib.load(open("./models/model_fit.pkl", "rb"))
print("Model Loaded")
imputer1 = joblib.load(open("./models/imputer.pkl", "rb"))
print("Model Loaded")
scaler = joblib.load(open("./models/scaler.pkl", "rb"))
print("Model Loaded")
encoder = joblib.load(open("./models/encoder.pkl", "rb"))
print("Model Loaded")
input_cols = joblib.load(open("./models/input_cols.pkl", "rb"))
print("Model Loaded")
target_col = joblib.load(open("./models/target_col.pkl", "rb"))
print("Model Loaded")
numeric_cols = joblib.load(open("./models/numeric_cols.pkl", "rb"))
print("Model Loaded")
categorical_cols = joblib.load(open("./models/categorical_cols.pkl", "rb"))
print("Model Loaded")
encoded_cols = joblib.load(open("./models/encoded_cols.pkl", "rb"))
print("Model Loaded")


@app.route("/",methods=['GET'])
@cross_origin()
def home():
	return render_template("index.html")

@app.route("/predict",methods=['GET', 'POST'])
@cross_origin()
def predict():
    if request.method == "POST":
		# DATE
        date = request.form['date']
        day = float(pd.to_datetime(date, format="%Y-%m-%dT").day)
        month = float(pd.to_datetime(date, format="%Y-%m-%dT").month)
        # MinTemp
        minTemp = float(request.form['mintemp'])
        # MaxTemp
        maxTemp = float(request.form['maxtemp'])
        # Rainfall
        rainfall = float(request.form['rainfall'])
        # Evaporation
        evaporation = float(request.form['evaporation'])
        # Sunshine
        sunshine = float(request.form['sunshine'])
        # Wind Gust Speed
        windGustSpeed = float(request.form['windgustspeed'])
        # Wind Speed 9am
        windSpeed9am = float(request.form['windspeed9am'])
        # Wind Speed 3pm
        windSpeed3pm = float(request.form['windspeed3pm'])
        # Humidity 9am
        humidity9am = float(request.form['humidity9am'])
        # Humidity 3pm
        humidity3pm = float(request.form['humidity3pm'])
        # Pressure 9am
        pressure9am = float(request.form['pressure9am'])
        # Pressure 3pm
        pressure3pm = float(request.form['pressure3pm'])
        # Temperature 9am
        temp9am = float(request.form['temp9am'])
        # Temperature 3pm
        temp3pm = float(request.form['temp3pm'])
        # Cloud 9am
        cloud9am = float(request.form['cloud9am'])
        # Cloud 3pm
        cloud3pm = float(request.form['cloud3pm'])
        # Cloud 3pm
        location = (request.form['location'])
        # Wind Dir 9am
        winddDir9am = (request.form['winddir9am'])
        # Wind Dir 3pm
        winddDir3pm = (request.form['winddir3pm'])
        # Wind Gust Dir
        windGustDir = (request.form['windgustdir'])
        # Rain Today
        rainToday = (request.form['raintoday'])

        input_lst = [location , minTemp , maxTemp , rainfall , evaporation , sunshine ,
					 windGustDir , windGustSpeed , winddDir9am , winddDir3pm , windSpeed9am , windSpeed3pm ,
					 humidity9am , humidity3pm , pressure9am , pressure3pm , cloud9am , cloud3pm , temp9am , temp3pm ,
					 rainToday , month , day]

        new_input = {
            'Location': location,
             'MinTemp': minTemp,
             'MaxTemp': maxTemp,
             'Rainfall': rainfall,
             'Evaporation': evaporation,
             'Sunshine': sunshine,
             'WindGustDir': windGustDir,
             'WindGustSpeed': windGustSpeed,
             'WindDir9am': winddDir9am,
             'WindDir3pm': winddDir3pm,
             'WindSpeed9am': windSpeed9am,
             'WindSpeed3pm': windSpeed3pm,
             'Humidity9am': humidity9am,
             'Humidity3pm': humidity3pm,
             'Pressure9am': pressure9am,
             'Pressure3pm': pressure3pm,
             'Cloud9am': cloud9am,
             'Cloud3pm': cloud3pm,
             'Temp9am': temp9am,
             'Temp3pm': temp3pm,
             'RainToday': rainToday,
             'Month' : month,
             'Day': day
        }

        def predict_input(input):
            input_df = pd.DataFrame([input])
            input_df[numeric_cols] = imputer1.transform(input_df[numeric_cols])
            input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
            input_df[encoded_cols] = encoder.transform(input_df[categorical_cols])
            X_input = input_df[numeric_cols + encoded_cols]
            pred = model_fit.predict(X_input)[0]
            prob = model_fit.predict_proba(X_input)[0][list(model_fit.classes_).index(pred)]
            return pred, prob
        pred= predict_input(new_input)
        output = pred
        #output1 = output[1]*100
        if output[0] == 'No':
                return render_template("after_sunny.html",output1=output[1]*100)
        elif output[0] == 'Yes':
                return render_template("after_rainy.html",output1=output[1]*100)
        else:
                return render_template()
    return render_template("predictor.html")

if __name__=='__main__':
	app.run(debug=True)