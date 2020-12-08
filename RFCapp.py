from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
from sklearn import preprocessing

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

label_encoder = preprocessing.LabelEncoder()

@app.route('/') # Homepage
def home():
    return render_template('main.html')


@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        state = request.form['STATE']
        district = request.form['DISTRICT']
        year = request.form['YEAR']
        data = pd.DataFrame([[state, district, year]],
                                       columns=['STATE', 'DISTRICT', 'YEAR'],
                                       dtype=int)
        data['STATE']= label_encoder.fit_transform(data['STATE'])
        data['DISTRICT']= label_encoder.fit_transform(data['DISTRICT'])
        prediction = model.predict(data)[0]
        L1 = prediction.tolist()
        return render_template('results.html',result=prediction, C1 = L1[0], C2 = L1[1], C3 = L1[2], C4 = L1[3], C5 = L1[4])

if __name__ == "__main__":
   app.run(debug=True)
