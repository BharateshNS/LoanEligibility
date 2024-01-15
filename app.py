# Import necessary libraries
from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)
model= pickle.load(open('RF_model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
         # Extract input values from the form
        gender = request.form['gender']
        dependents = request.form['dependents']
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
        dependents_mapping = {"1": 1, "2": 2, "3+": 3}  # Define the mapping
        # Convert dependents to an integer using the mapping
        dependents = dependents_mapping.get(dependents, 0)

        # Make a prediction using the loaded model
        features = np.array([[gender, married,dependents, education, self_employed,
                              applicant_income, coapplicant_income,
                              loan_amount, loan_amount_term, credit_card]])
        prediction = model.predict(features)
        
        # Interpret the prediction result
        result = "Yes you can Apply" if prediction[0] == 1 else "No you can't Apply"

    # Render the result template with the original form values
    return render_template('results.html', form_values=request.form, result=result)

# New route for the home page after prediction
@app.route('/home')
def go_home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
