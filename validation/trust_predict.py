import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
import math


class TrustTransfer_(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TrustTransfer_, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x

    def fit(self, x_train, y_train, num_epochs=300, learning_rate=0.01, verbose=True):
        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9)
        y_train = y_train.unsqueeze(1)
        for epoch in range(num_epochs):
            outputs = self(x_train)
            loss = criterion(outputs, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if verbose and (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    def test_model(self, x_test, y_test):
        with torch.no_grad():
            self.eval()
            predictions = self(x_test)
            predictions = torch.clamp(predictions, 0.0, 1.0)
            predictions = predictions.squeeze()
        mse = mean_squared_error(predictions.cpu().numpy(), y_test.cpu().numpy())
        rmse = math.sqrt(mse)
        std = torch.std(predictions).item()
        return predictions, rmse, std