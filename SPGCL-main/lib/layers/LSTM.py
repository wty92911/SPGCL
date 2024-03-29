import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))  
        out = self.fc(out[:, :, :])  
        return out
    
def train(trainX, trainY, num_nodes, device, hidden_size = 1000, num_layers = 3, learning_rate = 1e-4, num_epochs = 5000):
    trainX = trainX.transpose(0, 2, 1)
    trainY = trainY.transpose(0, 2, 1)
    trainX = torch.tensor(trainX).to(device)
    trainY = torch.tensor(trainY).to(device)
    model = LSTM(num_nodes, hidden_size, num_layers, num_nodes).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training
    for epoch in range(num_epochs):
        model.train()
        # X = trainX.reshape(-1, num_nodes * seq_len)
        outputs = model(trainX)
        optimizer.zero_grad()
        
        # obtain the loss function
        loss = criterion(outputs, trainY)
        
        loss.backward()
        
        optimizer.step()
        
        if epoch % 100 == 0:
            print('LSTM Epoch: %d, loss: %1.5f' % (epoch, loss.item()))
    return model
def predict(model, testX, device):
    testX = testX.transpose(0, 2, 1)
    model.eval()
    X = torch.tensor(testX).to(device)
    Y = model(X)
    Y = Y.cpu().detach().numpy().transpose(0, 2, 1)
    return Y