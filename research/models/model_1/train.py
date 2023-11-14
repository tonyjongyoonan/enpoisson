import torch
import torch.optim as optim
from models import MyModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

def train(device, model, train_loader, num_epochs):
    writer = SummaryWriter()
    for epoch in range(num_epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            writer.add_scalar("Loss/train", loss, epoch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    writer.flush()
    writer.close()
if __name__ == "__main__":
    #TODO: Import your dataset here

    # Initialize model, loss function, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyModel()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # DataLoader for your dataset
    train_loader = DataLoader(YOUR_DATASET, batch_size=BATCH_SIZE, shuffle=True)

    # Train the model
    train(device, model, train_loader, NUM_EPOCHS)

    # Save the model
    torch.save(model.state_dict(), 'my_ml_project/models/model_1/saved_model/model_state.pth')