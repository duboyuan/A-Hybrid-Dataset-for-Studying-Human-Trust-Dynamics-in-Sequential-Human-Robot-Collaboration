import math

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error


class TrustTransfer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TrustTransfer, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = nn.Sigmoid()(x)
        return x

    def ln_trans(self, trust):
        ln_result = []
        for index,x in enumerate(trust):
            if trust[index] == 0:
                x=1
            ln_result.append(-math.log(x/100))
        return ln_result

    def train_trust_transfer(self,x_train, y_train, num_epochs=300, learning_rate=0.01):
        criterion = nn.MSELoss()
        # criterion = torch.nn.BCELoss(reduction='mean')
        optimizer = optim.SGD(self.parameters(), lr=learning_rate,momentum=0.9)
        y_train = y_train.unsqueeze(1)
        for epoch in range(num_epochs):
            outputs = self(x_train)
            loss = criterion(outputs, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():}')

    def test_model(self, x_test, y_test):
        #y_test = y_test.unsqueeze(1)
        y_test = y_test
        with torch.no_grad():
            self.eval()
            predictions = self(x_test)
            # self.train()
            predictions = torch.round(torch.clamp(predictions, 0.0, 1.0),decimals=2)
            predictions = predictions.squeeze()
        mse = mean_squared_error(predictions,y_test)
        rmse = torch.sqrt(nn.functional.mse_loss(predictions, y_test))
        std = torch.std(predictions)

        return predictions, rmse, std

