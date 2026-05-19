# Project: Iris Flower Classification using Decision Tree
# Goal: Predict flower species based on sepal and petal values
# 1. Import required libraries
import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 2. Load dataset
df_iris = pd.read_csv("C:/Users/HP/3D Objects/Desktop/DA_using_Python/ML/(3) Decision Tree/(1) Iris.csv")
print(df_iris)
print("Shape of Dataset:", df_iris.shape)
print(df_iris.info())
print(df_iris.isnull().sum())   # Check missing values


# 3. Convert categorical Species column into numeric values
    # ML models work better with numeric data
le = LabelEncoder()
# Example:  Iris-setosa-> 0     Iris-versicolor -> 1        Iris-virginica  -> 2
df_iris['Species_new'] = le.fit_transform(df_iris['Species'])
print(df_iris)

# 4. Drop unnecessary columns
# Drop original Species text column
df_iris = df_iris.drop('Species', axis='columns')
# Drop Id column (not useful for prediction)
df_iris = df_iris.drop('Id', axis='columns')
print(df_iris)

# 5. Separate independent and dependent variables
df_input = df_iris.drop('Species_new', axis='columns')  # Input features
df_output = df_iris['Species_new']  # Output target

# 6. Create Decision Tree model
model = tree.DecisionTreeClassifier()
model.fit(df_input, df_output)  # Train model
print("Model Trained Successfully")

# 7. Direct prediction
    # Input format: [SepalLength, SepalWidth, PetalLength, PetalWidth]
print("Prediction:", model.predict([[4.9, 2.5, 4.5, 1.7]]))

# 8. Check full model accuracy
print("Overall Accuracy:", model.score(df_input, df_output))

# 9. Split data into training and testing sets
x = df_input
y = df_output
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print("Training Records:", len(x_train))
print("Testing Records:", len(x_test))

# 10. Predict using training data
y_pred_train = model.predict(x_train)
train_acc = accuracy_score(y_train, y_pred_train)
print("Training Accuracy:", train_acc)

# 11. Predict using testing data
y_pred_test = model.predict(x_test)
test_acc = accuracy_score(y_test, y_pred_test)
print("Testing Accuracy:", test_acc)

# 12. Another sample prediction
print(model.predict([[6.3, 2.3, 4.4, 1.3]]))

# 13. User input prediction
spl = float(input("Enter Sepal Length (cm): "))
spw = float(input("Enter Sepal Width (cm): "))
ppl = float(input("Enter Petal Length (cm): "))
ppw = float(input("Enter Petal Width (cm): "))
result = model.predict([[spl, spw, ppl, ppw]])

# 14. Convert numeric output to flower name
if result[0] == 0:
    print("Type of Flower is Iris-setosa")
elif result[0] == 1:
    print("Type of Flower is Iris-versicolor")
else:
    print("Type of Flower is Iris-virginica")