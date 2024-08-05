import requests

# Replace with your API endpoint
url = 'http://ec2-54-235-39-97.compute-1.amazonaws.com:8080/prediction'

# If you need to send data, include it here (for POST, PUT requests, etc.)
data = {
  "Gender": 1,  "Married": 1,  "Dependents": 1,  "Education": 1,  "Self_Employed": 0,  "LoanAmount": 120,
  "Loan_Amount_Term": 360,  "Credit_History": 0,  "Property_Area": 0,  "TotalIncome": 3000}

# For a POST request
response = requests.post(url, json=data)

# Check the response status code
if response.status_code == 200:
    # The request was successful, process the response data
    result = response.json()  # If the response is in JSON format
    print(result)
else:
    # Handle the error
    print(f"Request failed with status code: {response.status_code}")
    print(response.text)


# screen -R deploy python3 app.py