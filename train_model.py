import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Create the dataset
data = {
    'age': [4, 5, 6, 7, 8, 9, 5, 7, 6, 4, 8, 10, 6, 7, 9, 5, 8, 10, 6, 7],
    'speech_rate': [80, 90, 100, 110, 85, 120, 95, 105, 130, 75, 115, 140, 85, 125, 135, 100, 110, 150, 95, 105],
    'pronunciation_difficulty': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    'sound_substitution': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    'word_repetition': [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
    'speech_pauses': [0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1],
    'nasal_speech': [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    'monotone_speech': [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1],
    'disorder': [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('speech_disorder_data.csv', index=False)


# Load dataset
df = pd.read_csv('speech_disorder_data.csv')

# Features and target variable
X = df[['age', 'speech_rate', 'pronunciation_difficulty', 'sound_substitution', 'word_repetition', 'speech_pauses', 'nasal_speech', 'monotone_speech']]
y = df['disorder']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.2f}')

# Save the model
joblib.dump(model, 'speech_disorder_model.pkl')
