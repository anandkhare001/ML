import unittest
import os
import pickle
import pytest
from prophet import Prophet

class TestClass(unittest.TestCase):
    # Get the current working directory
    CURRENT_DIRECTORY = os.getcwd()

    def test_loan_pred_model(self):
        # Load Model
        model_path = os.path.join(self.CURRENT_DIRECTORY, 'LoanPredictor', 'RF_Loan_Model.pkl')
        model = pickle.load(open(model_path, 'rb'))
        data = {
                "Gender": 1,  "Married": 1,  "Dependents": 1,  "Education": 1,  "Self_Employed": 0,  "LoanAmount": 120,
                "Loan_Amount_Term": 360,  "Credit_History": 0,  "Property_Area": 0,  "TotalIncome": 3000
                }
        
        # Predict Loan Approval
        prediction = model.predict([[
                                    data['Gender'], data['Married'], data['Dependents'], data['Education'], data['Self_Employed'],
                                    data['LoanAmount'], data['Loan_Amount_Term'], data['Credit_History'], data['Property_Area'], data['TotalIncome']
                                    ]])

        # Do assertions
        self.assertFalse(prediction)
    
    def test_rainfall_trends(self):
        # Load Model
        model_path = os.path.join(self.CURRENT_DIRECTORY, 'Rainfall_Trends', 'model.pkl')
        model = pickle.load(open(model_path, 'rb'))

        # create a future dataframe for the next 20 years
        future = model.make_future_dataframe(periods=20, freq='YE')        
        forecast = model.predict(future)

        # Do assertions
        self.assertIsInstance(model, Prophet)
        self.assertEqual(future.ds[0], forecast.ds[0])
        self.assertEqual(future.ds[0], forecast.ds[0])
        self.assertEqual(forecast.trend[0], 1040.6390576)