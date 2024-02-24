import torch
import torch.nn as nn
import torch.nn.functional as F


class ChessCNN_no_pooling(nn.Module):
    def __init__(self, d_out):
        super(ChessCNN_no_pooling, self).__init__()
        # Assuming each channel represents a different piece type (e.g., 6 channels for 6 types each)
        self.conv1 = nn.Conv2d(12, 36, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(36, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # Assuming an 8x8 chess board
        self.fc2 = nn.Linear(128, d_out)

    def forward(self, x):
        # Apply convolutions
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Flatten the tensor
        x = x.view(x.size(0), -1)

        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class ChessCNN_pooling(nn.Module):
    def __init__(self, d_out):
        super(ChessCNN_pooling, self).__init__()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)  # Pooling layer to reduce spatial dimensions
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        # Adjusted size for the fully connected layer due to pooling
        self.fc1 = nn.Linear(128 * 4 * 4, 128)  # Reduced size due to pooling
        self.fc2 = nn.Linear(128, d_out)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)  # Apply pooling again to reduce size further
        x = x.view(x.size(0), -1)  # Flatten the tensor

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DenseLayer(nn.Module):
    def __init__(self, input_features, growth_rate):
        super(DenseLayer, self).__init__()
        self.bn = nn.BatchNorm2d(input_features)
        self.conv = nn.Conv2d(
            input_features, growth_rate, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.se = SELayer(growth_rate)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = self.se(out)
        return torch.cat([x, out], 1)


class DenseBlock(nn.Module):
    def __init__(self, num_layers, input_features, growth_rate):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(input_features + i * growth_rate, growth_rate))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class TransitionLayer(nn.Module):
    def __init__(self, input_features, output_features):
        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(input_features)
        self.conv = nn.Conv2d(
            input_features, output_features, kernel_size=1, stride=1, bias=False
        )

    def forward(self, x):
        return self.conv(F.relu(self.bn(x)))


class DenseNetEncoder(nn.Module):
    def __init__(self, d_out):
        super(DenseNetEncoder, self).__init__()
        # Initial convolution
        self.init_conv = nn.Conv2d(
            6, 64, kernel_size=3, stride=1, padding=1, bias=False
        )

        # Dense Blocks and Transition Layers with SE Blocks
        self.dense_block1 = DenseBlock(num_layers=4, input_features=64, growth_rate=12)
        self.trans_layer1 = TransitionLayer(
            input_features=64 + 4 * 12, output_features=64
        )
        self.dense_block2 = DenseBlock(num_layers=4, input_features=64, growth_rate=12)
        self.trans_layer2 = TransitionLayer(
            input_features=64 + 4 * 12, output_features=64
        )
        self.dense_block3 = DenseBlock(num_layers=4, input_features=64, growth_rate=12)

        # Global Pooling and Dense Layer for encoding
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 + 4 * 12, d_out)

    def forward(self, x):
        out = self.init_conv(x)
        out = self.dense_block1(out)
        out = self.trans_layer1(out)
        out = self.dense_block2(out)
        out = self.trans_layer2(out)
        out = self.dense_block3(out)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def train_cnn(
    device,
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs,
    learn_decay,
):
    train_loss_values = []
    train_error = []
    val_loss_values = []
    val_error = []
    val_3_accuracy = []
    swa_model = AveragedModel(model)
    swa_start = 1
    for epoch in range(num_epochs):
        train_correct = 0
        train_total = 0
        training_loss = 0.0
        # Training
        model.train()
        count = 0
        for sequences, labels in train_loader:
            count += 1
            sequences, labels = sequences.to(device), labels.to(device)
            # Forward Pass
            output = model(sequences)
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
            if count % 1000 == 0:
                print(
                    f"Epoch {epoch+1}, Batch: {count}| Training Loss: {training_loss/count}"
                )
        if epoch >= swa_start:
            swa_model.update_parameters(model)
        torch.optim.swa_utils.update_bn(train_loader, swa_model)
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        validation_loss = 0.0
        if val_loader is not None:
            with torch.no_grad():
                val_correct = 0
                val_total = 0
                val_top3_correct = 0
                validation_loss = 0

                for sequences, labels in val_loader:
                    sequences, labels = sequences.to(device), labels.to(device)
                    outputs = model(sequences)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    val_top3_correct += top_3_accuracy(labels, outputs) * labels.size(0)
                    loss = criterion(outputs, labels)
                    validation_loss += loss.item()

                val_loss_values.append(validation_loss / len(val_loader))
                val_accuracy = 100 * val_correct / val_total
                val_top3_accuracy = 100 * val_top3_correct / val_total
                val_error.append(100 - val_accuracy)
                val_3_accuracy.append(val_top3_accuracy)

        # Log Model Performance
        train_loss_values.append(training_loss)
        train_error.append(100 - 100 * train_correct / train_total)
        print(
            f"Epoch {epoch+1}, Training Loss: {training_loss/len(train_loader)}, Validation Error: {val_error[-1]}, Validation Top-3 Accuracy: {val_3_accuracy[-1]}, Training Error: {train_error[-1]}"
        )
        for op_params in optimizer.param_groups:
            op_params["lr"] = op_params["lr"] * learn_decay
    return train_error, train_loss_values, val_error, val_loss_values, swa_model
