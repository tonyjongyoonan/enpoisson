import torch
import torch.nn as nn
import torch.optim as optim
from .model import ChessAutoEncoder
import matplotlib.pyplot as plt
from ...parse_data import get_training_data_raw, transform_data
from torch.utils.data import Dataset, DataLoader

NUM_EPOCHS = 30

class MyDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
def train(device, model, train_loader, val_loader, num_epochs):
    train_loss_values = []
    train_error = []
    val_loss_values = []
    val_error = []
    for epoch in range(num_epochs):
        train_correct = 0
        train_total = 0
        training_loss = 0.0
        for op_params in optimizer.param_groups:
            op_params['lr'] = op_params['lr'] * 0.99
        # Training
        model.train()
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            # Forward Pass
            output = model(data)
            loss = criterion(output, labels)
            # Backpropogate & Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # For logging purposes
            training_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        validation_loss = 0.0
        if val_loader is not None:
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    loss = criterion(outputs, labels)
                    validation_loss += loss.item()
            val_loss_values.append(validation_loss / len(val_loader))
            val_error.append(100-100*val_correct/val_total)
        # Log Model Performance  
        train_loss_values.append(training_loss)
        train_error.append(100-100*train_correct/train_total)
        print(f'Epoch {epoch+1}, Training Loss: {training_loss}, Validation Error: {val_error[-1]}, Error: {train_error[-1]}')
    return train_error,train_loss_values, val_error, val_loss_values

if __name__ == "__main__":
    #Import dataset and generate data loaders
    filepath = "/Users/jlee0/Desktop/cis400/enpoisson/lichess_db_standard_rated_2013-01.pgn"
    raw_data = get_training_data_raw(50)
    X,Y = transform_data(raw_data)
    data = MyDataset(X,Y)
    train_loader = DataLoader(data, batch_size=32, shuffle=True)
    raw_val_data = get_training_data_raw(10)
    X_val,Y_val = transform_data(raw_val_data)
    val_data = MyDataset(X_val,Y_val)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=True)
    # Initialize model, loss function, and optimizer
    feature_length = len(X[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessAutoEncoder(feature_length)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Train the model
    train_error,train_loss_values, val_error, val_loss_values = train(device, model, train_loader, val_loader, NUM_EPOCHS)

    # Plot the training error
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_values, label='Training Loss')
    plt.xlabel('Weight Updates')
    plt.ylabel('Error')
    plt.title('Training Error')
    plt.legend()
    plt.show()
    plt.savefig('training_error_model_autoencoder.png')  # This will save the plot as an image

    # Save the model
    torch.save(model.state_dict(), 'model_state.pth')