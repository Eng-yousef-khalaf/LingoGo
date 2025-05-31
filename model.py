from sklearn.ensemble import RandomForestRegressor
import pickle
import pandas as pd

# Example data including the new features
data = {
    'num_presentations': [5, 3, 7],
    'num_levels': [2, 4, 6],
    'total_quiz_score': [80, 75, 90],
    'avg_quiz_score': [85, 80, 88],
    'num_slides': [12, 10, 14],
    'unique_topics': [1, 2, 3],  # New feature
    'language_pair': [0, 1, 0],  # New feature
    'overall': [80, 70, 90]  # Target variable
}

df = pd.DataFrame(data)

# Features and target
X = df[['num_presentations', 'num_levels', 'total_quiz_score', 'avg_quiz_score', 'num_slides', 'unique_topics', 'language_pair']]
y = df['overall']

# Train the model
model = RandomForestRegressor()
model.fit(X, y)

# Save the model to a file
with open("user_performance_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as 'user_performance_model.pkl'!")
