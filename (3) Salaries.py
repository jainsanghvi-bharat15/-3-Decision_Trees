# 1. Import required libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

# 2. Load dataset (Salary classification data)
df = pd.read_csv("C:/Users/HP/3D Objects/Desktop/DA_using_Python/ML/(3) Decision Tree/(3) Salaries.csv")

# Display basic information
print(df.head())          # First 5 rows
print(df.shape)           # (rows, columns)
print(df.isnull().sum())  # Check missing values

# 3. Separate independent (input) and dependent (output) variables
    # Input features → company, job, degree
df_input = df.drop('salary', axis='columns')
print("\nInput Data:\n", df_input)

    # Output → salary (0 or 1)
df_output = df['salary']
print("\nOutput Data:\n", df_output)

# 4. Convert categorical (string) data into numeric using LabelEncoder
    # Machine Learning models cannot work with text directly
lb_com = LabelEncoder()     # For 'company'
lb_job = LabelEncoder()     # For 'job'
lb_degree = LabelEncoder()  # For 'degree'

# Convert text → numeric values
df_input['company_new'] = lb_com.fit_transform(df_input['company'])
df_input['job_new'] = lb_job.fit_transform(df_input['job'])
df_input['degree_new'] = lb_degree.fit_transform(df_input['degree'])

# Drop original string columns
df_input.drop(['company', 'job', 'degree'], axis='columns', inplace=True)
print("\nEncoded Input Data:\n", df_input)

# 5. Create and train Decision Tree model
model = tree.DecisionTreeClassifier()
model.fit(df_input, df_output)
print("\nModel Trained:", model)

# 6. Make prediction
    # Input format: [company_new, job_new, degree_new]  Example: [2,1,2]
    # Output: 0 = Low salary, 1 = High salary
print("\nPrediction:", model.predict([[2, 1, 2]]))

# 7. Check model accuracy
print("Model Accuracy:", model.score(df_input, df_output))