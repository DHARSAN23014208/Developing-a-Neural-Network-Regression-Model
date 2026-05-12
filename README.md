# NAME: DHARSAN KUMAR R
# REG.NO: 212223240028
# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
The objective of this experiment is to design, implement, and evaluate a Deep Learning–based Neural Network regression model to predict a continuous output variable from a given set of input features. The task is to preprocess the data, construct a neural network regression architecture, train the model using backpropagation and gradient descent, and evaluate its performance using appropriate regression metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R² score.


## Neural Network Model
<img width="1060" height="665" alt="image" src="https://github.com/user-attachments/assets/645208f1-10fd-4460-82bc-88bdd9f84b28" />


## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name:DHARSAN KUMAR R

### Register Number:212223240028

```
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load Dataset
dataset1 = pd.read_csv('C:\\Users\\admin\\Desktop\\DEEPLEARNING\\DEEPLEARNING - Sheet1.csv')

X = dataset1[['Input']].values
y = dataset1[['Output']].values

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=33
)

# Feature Scaling
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)


class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(1, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
        
        self.history = {'loss': []}

    def forward(self, x):
        return self.model(x)


ai_brain = NeuralNet()

criterion = nn.MSELoss()
optimizer = optim.Adam(ai_brain.parameters(), lr=0.01)


def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    
    for epoch in range(epochs):
        
       
        outputs = ai_brain(X_train)
        loss = criterion(outputs, y_train)
        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    
        ai_brain.history['loss'].append(loss.item())
        
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')


train_model(ai_brain, X_train_tensor, y_train_tensor, criterion, optimizer)


with torch.no_grad():
    test_loss = criterion(ai_brain(X_test_tensor), y_test_tensor)
    print(f'Test Loss: {test_loss.item():.6f}')


loss_df = pd.DataFrame(ai_brain.history)

loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()


X_n1_1 = torch.tensor([[9]], dtype=torch.float32)

prediction = ai_brain(
    torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)
).item()

print(f'Prediction: {prediction}')

```

### Dataset Information
### OUTPUT

<img width="180" height="263" alt="image" src="https://github.com/user-attachments/assets/5b85226c-7b25-445a-b720-7003c9b4e7d2" />

### Training Loss Vs Iteration Plot

<img width="580" height="455" alt="image" src="https://github.com/user-attachments/assets/ca09acdf-fc88-42a4-898e-2afa9c3c33e7" />

### New Sample Data Prediction

<img width="311" height="42" alt="image" src="https://github.com/user-attachments/assets/1bada11f-8bfd-404e-9f6b-1e45feeb092e" />

## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
