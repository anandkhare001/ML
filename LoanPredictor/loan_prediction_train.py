# Importing the required packages
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
import pickle



# load the dataset
dataset = pd.read_csv("train.csv")
numerical_cols = dataset.select_dtypes(include=['int64','float64']).columns.tolist()
categorical_cols = dataset.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('Loan_Status')
categorical_cols.remove('Loan_ID')

# Filling categorical columns with mode
for col in categorical_cols:
    dataset[col].fillna(dataset[col].mode()[0], inplace=True)

# Filling Numerical columns with median
for col in numerical_cols:
    dataset[col].fillna(dataset[col].median(), inplace=True)

# Take care of outliers
dataset[numerical_cols] = dataset[numerical_cols].apply(lambda x: x.clip(*x.quantile([0.05, 0.95])))

# Log Transforamtion & Domain Processing
dataset['LoanAmount'] = np.log(dataset['LoanAmount']).copy()
dataset['TotalIncome'] = dataset['ApplicantIncome'] + dataset['CoapplicantIncome']
dataset['TotalIncome'] = np.log(dataset['TotalIncome']).copy()


# Dropping ApplicantIncome and CoapplicantIncome
dataset = dataset.drop(columns=['ApplicantIncome','CoapplicantIncome'])

# Label encoding categorical variables
for col in categorical_cols:
    le = LabelEncoder()
    dataset[col] = le.fit_transform(dataset[col])

#Encode the target columns
dataset['Loan_Status'] = le.fit_transform(dataset['Loan_Status'])

# Train test split
X = dataset.drop(columns=['Loan_Status', 'Loan_ID'])
y = dataset.Loan_Status
RANDOM_SEED = 6

#X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =0.3, random_state = RANDOM_SEED)

# RandomForest
rf = RandomForestClassifier(random_state=RANDOM_SEED)
param_grid_forest = {
    'n_estimators': [200,400, 700],
    'max_depth': [10,20,30],
    'criterion' : ["gini", "entropy"],
    'max_leaf_nodes': [50, 100]
}

grid_forest = GridSearchCV(
        estimator=rf,
        param_grid=param_grid_forest, 
        cv=5, 
        n_jobs=-1, 
        scoring='accuracy',
        verbose=0
    )
model_forest = grid_forest.fit(X, y)


filename = 'RF_Loan_model.pkl'
pickle.dump(model_forest, open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))

data = [[
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                4.98745,
                360.0,
                1.0,
                2.0,
                8.698
            ]]
print(f"Prediction is : {loaded_model.predict(pd.DataFrame(data))}")