# Titanic Movie
# Import required libraries
import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load Titanic dataset
df = pd.read_csv("C:/Users/HP/3D Objects/Desktop/DA_using_Python/ML/(3) Decision Tree/(5) Titanic.csv")
print(df.head(15))  # Display first 15 rows
print("Shape of Dataset:", df.shape)
print(df.info())


# Check missing values before cleaning
print("Missing Values Before Cleaning:\n", df.isnull().sum())
df.dropna(inplace=True) # Remove rows containing missing values
# Check missing values after cleaning
print("Missing Values After Cleaning:\n", df.isnull().sum())

# Convert categorical column 'Sex' into numeric values
    # male = 1, female = 0 (depends on LabelEncoder mapping)
S = LabelEncoder()
df['Sex_New'] = S.fit_transform(df['Sex'])
print(df.head(5))

# Drop original text column
df.drop('Sex', axis='columns', inplace=True)
print(df.head(5))

# Separate input and output data
df_input = df.drop('Survived', axis='columns') # Input features: Pclass, Sex_New, Age, Fare
df_output = df['Survived']  # Output target: Survived (0 = No, 1 = Yes)

model = tree.DecisionTreeClassifier()   # Create Decision Tree model
model.fit(df_input, df_output)          # Train model
print("Model trained successfully")

Class = int(input("Enter Passenger Class (1,2,3): "))   # Take user input
if Class < 1 or Class > 3:
    print("Invalid Class")

Sex = int(input("Enter Sex (1 = Male, 0 = Female): "))
if Sex < 0 or Sex > 1:
    print("Invalid Sex")
Age = int(input("Enter Age: "))
Fare = float(input("Enter Fare: "))
pred = model.predict([[Class, Sex, Age, Fare]]) # Predict survival
print("Prediction:", pred)

# Show result
if pred[0] == 0:
    print("Passenger Not Survived")
else:
    print("Passenger Survived")
 
# With Training and Testing Data
# Store into X and Y
x = df_input
y = df_output

# Split dataset into training and testing data
# 50% training and 50% testing
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.5, random_state=42)
print("Length of Training Dataset:", len(xtrain))
print("Length of Testing Dataset:", len(xtest))
model = tree.DecisionTreeClassifier()   # Create Decision Tree model
model.fit(xtrain, ytrain)   # Train model
print("\nModel Trained Successfully:", model)
y_pred_train = model.predict(xtrain)    # Prediction on training data
train_acc = accuracy_score(ytrain, y_pred_train)    # Accuracy on training data
print("Training Accuracy:", train_acc)

# Prediction on testing data
y_pred_test = model.predict(xtest)  
print("Predicted Testing Values:\n", y_pred_test)

# Accuracy on testing data
test_acc = accuracy_score(ytest, y_pred_test)
print("Testing Accuracy:", test_acc)