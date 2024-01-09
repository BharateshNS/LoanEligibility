# import packges
import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app =Flask(__name__)
model= pickle.load(open('RF_model.pkl','rb'))

@app.route('/')

def home():
    return render_template('index.html')

@app.route('/predict',methods= ['POST'])

def predict():
    if request.method == 'POST':
         # Extract input values from the form
        gender = request.form['gender']
        gender = request.form['gender']
        married = request.form['married']
        education = request.form['education']
        self_employed = request.form['self_employed']
        applicant_income = float(request.form['applicant_income'])
        coapplicant_income = float(request.form['coapplicant_income'])
        loan_amount = float(request.form['loan_amount'])
        loan_amount_term = float(request.form['loan_amount_term'])
        credit_card = request.form['credit_card']

        # Preprocess categorical variables
        gender = 1 if gender == 'Female' else 0
        married = 1 if married == 'Yes' else 0
        education = 1 if education == 'Graduate' else 0
        self_employed = 1 if self_employed == 'Yes' else 0
        credit_card = 1 if credit_card == 'Yes' else 0

        # Make a prediction using the loaded model
        features = np.array([[gender, married, education, self_employed,
                              applicant_income, coapplicant_income,
                              loan_amount, loan_amount_term, credit_card]])
        prediction = model.predict(features)
# Interpret the prediction result
        result = "Yes you can Apply for Loan" if prediction[0] == 1 else "No you can't Apply for Loan" 

        return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)