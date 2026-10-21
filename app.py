import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load the trained model
with open('loan_status_logistic_regression.pkl', 'rb') as f:
    model = pickle.load(f)

# Attractive HTML layout with embedded Tailwind CSS styling
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Loan Eligibility Predictor</title>
    <script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; }
    </style>
</head>
<body class="bg-gradient-to-br from-slate-900 via-indigo-950 to-slate-900 min-h-screen text-slate-100 flex items-center justify-center p-4">
    <div class="max-w-2xl w-full bg-slate-800/80 backdrop-blur-md rounded-2xl shadow-2xl border border-slate-700 overflow-hidden p-8">
        <div class="text-center mb-8">
            <h1 class="text-3xl font-bold bg-gradient-to-r from-blue-400 to-emerald-400 bg-clip-text text-transparent">Loan Eligibility Predictor</h1>
            <p class="text-slate-400 mt-2 text-sm">Fill in your financial and personal details below to check your loan approval status.</p>
        </div>

        <form action="/predict" method="POST" class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
                <label class="block text-sm font-medium text-slate-300 mb-1">Applicant Income ($)</label>
                <input type="number" step="any" name="applicantincome" required placeholder="e.g. 5000" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-white focus:outline-none focus:border-blue-500 transition">
            </div>

            <div>
                <label class="block text-sm font-medium text-slate-300 mb-1">Coapplicant Income ($)</label>
                <input type="number" step="any" name="coapplicantincome" required placeholder="e.g. 1500" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-white focus:outline-none focus:border-blue-500 transition">
            </div>

            <div>
                <label class="block text-sm font-medium text-slate-300 mb-1">Loan Amount ($)</label>
                <input type="number" step="any" name="loanamount" required placeholder="e.g. 150" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-white focus:outline-none focus:border-blue-500 transition">
            </div>

            <div>
                <label class="block text-sm font-medium text-slate-300 mb-1">Loan Term (Days)</label>
                <input type="number" name="loan_amount_term" value="360" required class="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-white focus:outline-none focus:border-blue-500 transition">
            </div>

            <div>
                <label class="block text-sm font-medium text-slate-300 mb-1">Credit History</label>
                <select name="credit_history" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-white focus:outline-none focus:border-blue-500 transition">
                    <option value="1">Yes (Good History)</option>
                    <option value="0">No (Bad History)</option>
                </select>
            </div>

            <div>
                <label class="block text-sm font-medium text-slate-300 mb-1">Gender</label>
                <select name="gender" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-white focus:outline-none focus:border-blue-500 transition">
                    <option value="Male">Male</option>
                    <option value="Female">Female</option>
                </select>
            </div>

            <div>
                <label class="block text-sm font-medium text-slate-300 mb-1">Married</label>
                <select name="married" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-white focus:outline-none focus:border-blue-500 transition">
                    <option value="Yes">Yes</option>
                    <option value="No">No</option>
                </select>
            </div>

            <div>
                <label class="block text-sm font-medium text-slate-300 mb-1">Education</label>
                <select name="education" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-white focus:outline-none focus:border-blue-500 transition">
                    <option value="Graduate">Graduate</option>
                    <option value="Not Graduate">Not Graduate</option>
                </select>
            </div>

            <div>
                <label class="block text-sm font-medium text-slate-300 mb-1">Self Employed</label>
                <select name="self_employed" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-white focus:outline-none focus:border-blue-500 transition">
                    <option value="No">No</option>
                    <option value="Yes">Yes</option>
                </select>
            </div>

            <div>
                <label class="block text-sm font-medium text-slate-300 mb-1">Property Area</label>
                <select name="property_area" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-white focus:outline-none focus:border-blue-500 transition">
                    <option value="Urban">Urban</option>
                    <option value="Semiurban">Semiurban</option>
                    <option value="Rural">Rural</option>
                </select>
            </div>

            <div class="md:col-span-2 mt-4">
                <button type="submit" class="w-full bg-gradient-to-r from-blue-500 to-emerald-500 text-white font-semibold py-3 rounded-lg shadow-lg hover:from-blue-600 hover:to-emerald-600 transition transform hover:-translate-y-0.5 cursor-pointer">
                    Predict Loan Status
                </button>
            </div>
        </form>

        {% if prediction_text %}
        <div class="mt-6 p-4 rounded-xl text-center border {% if 'Approved' in prediction_text %} bg-emerald-950/50 border-emerald-500/50 text-emerald-300 {% else %} bg-rose-950/50 border-rose-500/50 text-rose-300 {% endif %} shadow-md">
            <p class="font-semibold text-lg">{{ prediction_text }}</p>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_PAGE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Fetch inputs from form
        applicantincome = float(request.form['applicantincome'])
        coapplicantincome = float(request.form['coapplicantincome'])
        loanamount = float(request.form['loanamount'])
        loan_amount_term = float(request.form['loan_amount_term'])
        credit_history = float(request.form['credit_history'])
        
        gender = request.form['gender']
        married = request.form['married']
        education = request.form['education']
        self_employed = request.form['self_employed']
        property_area = request.form['property_area']

        # Reconstruct DataFrame matching the exact features expected by the model
        input_data = pd.DataFrame(0, index=[0], columns=model.feature_names_in_)

        # Map base features
        if 'applicantincome' in input_data.columns:
            input_data['applicantincome'] = applicantincome
        if 'coapplicantincome' in input_data.columns:
            input_data['coapplicantincome'] = coapplicantincome
        if 'loanamount' in input_data.columns:
            input_data['loanamount'] = loanamount
        if 'loan_amount_term' in input_data.columns:
            input_data['loan_amount_term'] = loan_amount_term
        if 'credit_history' in input_data.columns:
            input_data['credit_history'] = credit_history

        # Map categorical variables
        if gender == 'Male' and 'gender_Male' in input_data.columns:
            input_data['gender_Male'] = 1
        if married == 'Yes' and 'married_Yes' in input_data.columns:
            input_data['married_Yes'] = 1
        if education == 'Not Graduate' and 'education_Not Graduate' in input_data.columns:
            input_data['education_Not Graduate'] = 1
        if self_employed == 'Yes' and 'self_employed_Yes' in input_data.columns:
            input_data['self_employed_Yes'] = 1
        if property_area == 'Semiurban' and 'property_area_Semiurban' in input_data.columns:
            input_data['property_area_Semiurban'] = 1
        elif property_area == 'Urban' and 'property_area_Urban' in input_data.columns:
            input_data['property_area_Urban'] = 1

        # Run Prediction
        prediction = model.predict(input_data)[0]
        
        if prediction == 'Y' or prediction == 1:
            result = "🎉 Congratulations! Your Loan Status is Approved."
        else:
            result = "❌ Sorry, your Loan Application is Rejected."

        return render_template_string(HTML_PAGE, prediction_text=result)

    except Exception as e:
        return render_template_string(HTML_PAGE, prediction_text=f"Error processing input: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
