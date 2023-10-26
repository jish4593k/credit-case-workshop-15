import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
from tensorflow import keras
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils import shuffle

# Load the dataset
data = pd.read_csv('./UCI_Credit_Card.csv')

# Data Exploration
print("Dataset Size:", data.shape)
print(data.describe())

# Explore the next month default payment status
next_month = data['default.payment.next.month'].value_counts()
print("Next Month Default Payment Status:")
print(next_month)

# Plot the next month default payment status
plt.figure(figsize=(6, 6))
plt.title('Credit Card Default Payment\n (Default: 1, Not Default: 0)')
sns.set_style("whitegrid")
sns.barplot(x=next_month.index, y=next_month.values)
plt.xlabel('Status')
plt.ylabel('Count')
plt.show()

# Feature selection: Remove ID and target column
data.drop(['ID'], inplace=True, axis=1)
target = data['default.payment.next.month'].values
features = data.drop(['default.payment.next.month'], axis=1).values

# Split the data into training and testing sets
train_x, test_x, train_y, test_y = train_test_split(features, target, test_size=0.30, stratify=target, random_state=1)

# Create a dictionary of classifiers and their parameter grids
classifiers = {
    'SVC': (SVC(), {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1]}),
    'DecisionTree': (DecisionTreeClassifier(), {'max_depth': [6, 9, 11]}),
    'RandomForest': (RandomForestClassifier(), {'n_estimators': [3, 5, 6]}),
    'KNeighbors': (KNeighborsClassifier(), {'n_neighbors': [4, 6, 8]})
}

# Perform GridSearchCV for each classifier
def grid_search_cv(pipeline, param_grid, train_x, train_y, test_x, test_y, score='accuracy'):
    response = {}
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring=score)
    search = grid_search.fit(train_x, train_y)
    print(f"{search.best_estimator_} - Best Score: {search.best_score_:.4f}")
    predict_y = grid_search.predict(test_x)
    accuracy = accuracy_score(test_y, predict_y)
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(test_y, predict_y))
    response['predict_y'] = predict_y
    response['accuracy_score'] = accuracy
    return response

for classifier_name, (classifier, param_grid) in classifiers.items():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        (classifier_name, classifier)
    ])
    print(f"GridSearchCV for {classifier_name}:")
    result = grid_search_cv(pipeline, param_grid, train_x, train_y, test_x, test_y, score='accuracy')

# Deep Learning with TensorFlow/Keras
model = keras.Sequential([
    keras.layers.Input(shape=(23,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_x, train_y, epochs=10, batch_size=32, validation_data=(test_x, test_y))

# Deep Learning with PyTorch
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init()
        self.fc1 = nn.Linear(23, 64)
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.sigmoid(self.fc2(x))
        return x

model = SimpleNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

train_x, train_y = shuffle(train_x, train_y)  # Shuffle data

train_x = torch.FloatTensor(train_x)
train_y = torch.FloatTensor(train_y).view(-1, 1)

test_x = torch.FloatTensor(test_x)
test_y = torch.FloatTensor(test_y).view(-1, 1)

train_dataset = TensorDataset(train_x, train_y)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {running_loss / (len(train_loader)):.4f}")

# Evaluate the PyTorch model
model.eval()
with torch.no_grad():
    outputs = model(test_x)
    predicted = (outputs > 0.5).float()
    accuracy = torch.sum(predicted == test_y) / len(test_y)
    print(f"PyTorch Model Accuracy: {accuracy.item():.4f}")
