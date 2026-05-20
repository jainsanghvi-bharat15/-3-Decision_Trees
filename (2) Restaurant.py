# Restaurant Dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df_rest = pd.read_csv("C:/Users/HP/3D Objects/Desktop/DA_using_Python/ML/(3) Decision Tree/(2) Restaurant.csv")
print(df_rest)
print("Shape of Dataset:", df_rest.shape)
print(df_rest.info())
print("Missing Values:\n", df_rest.isnull().sum())  # Check missing values

# Label Encoding
# Convert text values into numeric values
le = LabelEncoder()
df_rest['Cuisine_new'] = le.fit_transform(df_rest['Cuisine'])
df_rest['MealTime_new'] = le.fit_transform(df_rest['MealTime'])
df_rest['AgeGroup_new'] = le.fit_transform(df_rest['AgeGroup'])
df_rest['Location_new'] = le.fit_transform(df_rest['Location'])

# Output column encoding
df_rest['RestaurantType'] = le.fit_transform(df_rest['RestaurantType'])
print(df_rest)

# Drop original text columns
df_rest = df_rest.drop(['Cuisine', 'MealTime', 'AgeGroup', 'Location'],
                        axis='columns')
print(df_rest.head())

# Define input and output variables
X = df_rest.drop('RestaurantType', axis=1)   # Features
y = df_rest['RestaurantType']                # Target
print("Input Data:\n", X)
print("Output Data:\n", y)

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))

# Create and train Decision Tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Accuracy on training data
print("Training Accuracy:", model.score(X_train, y_train))

# Prediction on testing data
y_pred = model.predict(X_test)
print("Testing Accuracy:", accuracy_score(y_test, y_pred))

# Detailed report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Train model on full dataset (optional)
model.fit(X, y)
print("Full Dataset Accuracy:", model.score(X, y))

# Example prediction
print("Sample Prediction:", model.predict([[1, 1, 2, 2]]))

# User Input Prediction
cus = int(input("Enter Cuisine Type (Indian-0, Chinese-1, Italian-2, Mexican-3): "))
mt = int(input("Enter Meal Time (Breakfast-0, Dinner-1, Lunch-2): "))
ag = int(input("Enter Age Group (Adult-0, Senior-1, Teen-2): "))
loc = int(input("Enter Location (Suburban-0, Rural-1, Urban-2): "))
result = model.predict([[cus, mt, ag, loc]])

# Display result
if result[0] == 0:
    print("Preferred Restaurant Type: Casual Dining")
elif result[0] == 1:
    print("Preferred Restaurant Type: Fast Food")
else:
    print("Preferred Restaurant Type: Fine Dining")