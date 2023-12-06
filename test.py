import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# Generate the time series dataset
dates = pd.date_range('2021-01-01', freq='D', periods=500)
X = np.random.normal(size=500)
Y = 2 * X + np.random.normal(size=500)

# Normalize the data
scaler = MinMaxScaler(feature_range=(-1, 1))
X_normalized = scaler.fit_transform(X.reshape(-1, 1))
Y_normalized = scaler.fit_transform(Y.reshape(-1, 1))

# Create sequences for the LSTM model
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        label = data[i+seq_length:i+seq_length+1]
        sequences.append((seq, label))
    return sequences

seq_length = 10  # You can adjust this based on your preference
X_sequences = create_sequences(X_normalized, seq_length)
Y_sequences = create_sequences(Y_normalized, seq_length)

# Convert sequences to PyTorch tensors
X_data = torch.tensor([seq[0] for seq in X_sequences], dtype=torch.float32)
Y_data = torch.tensor([seq[1] for seq in Y_sequences], dtype=torch.float32)

# Split the dataset into training and testing sets
train_size = int(len(X_sequences) * 0.8)
test_size = len(X_sequences) - train_size
X_train, X_test = X_data[:train_size], X_data[train_size:]
Y_train, Y_test = Y_data[:train_size], Y_data[train_size:]

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])  # Use the output from the last time step
        return out

# Instantiate the model
input_size = 1  # Number of features in the input
hidden_size = 64  # Number of hidden units in the LSTM
output_size = 1  # Number of output units (for regression)
model = LSTMModel(input_size, hidden_size, output_size)

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 100
batch_size = 32
train_dataset = TensorDataset(X_train, Y_train)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    for batch_X, batch_Y in train_dataloader:
        # Forward pass
        outputs = model(batch_X.unsqueeze(-1))  # Add an extra dimension for the input sequence

        # Compute the loss
        loss = criterion(outputs, batch_Y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Test the model
model.eval()
with torch.no_grad():
    test_outputs = model(X_test.unsqueeze(-1))  # Add an extra dimension for the input sequence
    test_loss = criterion(test_outputs, Y_test)
    test_outputs = scaler.inverse_transform(test_outputs.numpy())
    Y_test = scaler.inverse_transform(Y_test.numpy())

# Print the test loss and predicted values
print(f'Test Loss: {test_loss.item():.4f}')
print('Predicted Values:', test_outputs.flatten())
print('True Values:', Y_test.flatten())
