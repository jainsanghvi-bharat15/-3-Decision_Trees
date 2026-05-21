# Project: Tennis Game Play
# Goal: Predict whether game will be played or not on the basis of atmosphere
# 1. Import required libraries
import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 2. Load dataset
df = pd.read_csv("C:/Users/HP/3D Objects/Desktop/DA_using_Python/ML/(3) Decision Tree/(4) Tennis.csv")
print(df)
print("Shape of Dataset:", df.shape)
print(df.info())
print(df.isnull().sum())   # Check missing values

# 3. Convert categorical Species column into numeric values
    # ML models work better with numeric data
OL = LabelEncoder()
TP = LabelEncoder()
HD = LabelEncoder()
WY = LabelEncoder()
PL = LabelEncoder()
df['Outlook_New'] = OL.fit_transform(df['outlook'])
df['Temp_New'] = TP.fit_transform(df['temp'])
df['Humid_New'] = HD.fit_transform(df['humidity'])
df['Windy_New'] = WY.fit_transform(df['windy'])
df['Play_New'] = PL.fit_transform(df['play'])
print(df)

# 4. Drop unnecessary columns
# Drop original Species text column
df.drop(['outlook','temp','humidity','windy','play'], axis='columns',inplace=True)
print(df)

# 5. Separate independent and dependent variables
df_input = df.drop('Play_New', axis='columns')  # Input features
df_output = df['Play_New']  # Output target

# 6. Create Decision Tree model
model = tree.DecisionTreeClassifier()
model.fit(df_input, df_output)  # Train model
print("Model Trained Successfully")

# 7. Direct prediction
    # Input format: [Outlook{0,1,2}, Temp{0,1,2}, Humidity{0,1}, Windy{0,1}]
print("If result is 0 means no game will be played, else 1.")
O=int(input("Enter about outlook (0 for overcast, 1 for rainy, 2 for sunny): "))
T=int(input("Enter about temperature (0 for cool, 1 for hot, 2 for mild): "))
H=int(input("Enter about humidity (0 for high, 1 for low): "))
W=int(input("Enter about wind (0 for false and 1 for true): "))
print("Prediction:", model.predict([[O,T,H,W]]))

# 8. Check full model accuracy
print("Overall Accuracy:", model.score(df_input, df_output))

# 9. Split data into training and testing sets
x = df_input
y = df_output
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)
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