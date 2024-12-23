import unittest
import os
import pickle
import pytest

class TestClass(unittest.TestCase):
    def test_loan_pred_model(self):
        model_path = os.path.join(os.getcwd(), '../LoanPredictor', 'RF_Loan_Model.pkl')
        model = pickle.load(open(model_path, 'rb'))
        data = {
                "Gender": 1,  "Married": 1,  "Dependents": 1,  "Education": 1,  "Self_Employed": 0,  "LoanAmount": 120,
                "Loan_Amount_Term": 360,  "Credit_History": 0,  "Property_Area": 0,  "TotalIncome": 3000
                }
        prediction = model.predict([[
                                    data['Gender'], data['Married'], data['Dependents'], data['Education'], data['Self_Employed'],
                                    data['LoanAmount'], data['Loan_Amount_Term'], data['Credit_History'], data['Property_Area'], data['TotalIncome']
                                    ]])

        self.assertFalse(prediction)
    
