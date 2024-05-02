import pickle
import torch
import pandas as pd
import importlib
import utils
import models
import data_processing_utils

importlib.reload(data_processing_utils)
from data_processing_utils import *
importlib.reload(utils)
from utils import *
importlib.reload(models)
from models import *
from torch.utils.data import DataLoader, Subset
import torch.optim as optim

def train_with_fen(device, model, train_loader, val_loader, criterion, optimizer, num_epochs, learn_decay, experiment_name):
    train_loss_values = []
    train_error = []
    val_loss_values = []
    val_error = []
    val_3_accuracy = []
    for epoch in range(num_epochs):
        train_correct = 0
        train_total = 0
        training_loss = 0.0
        # Training
        model.train()
        count = 0
        for boards, sequences, lengths, fens, labels in train_loader:
            count += 1
            boards, sequences, lengths, labels = boards.to(device, non_blocking=True), sequences.to(device, non_blocking=True), lengths, labels.to(device, non_blocking=True)
            # Forward Pass
            output = model(boards, sequences, lengths)
            loss = criterion(output, labels)
            # Backpropogate & Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # For logging purposes
            training_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += labels.size(0)
            batch_correct = (predicted == labels).sum().item()
            train_correct += batch_correct
            if count % 1000 == 0:
                print(f'Epoch {epoch+1}, Batch: {count}| Training Loss: {loss.item()} | Training Error: {batch_correct/labels.size(0)}')
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

                for boards, sequences, lengths, fens, labels in val_loader:
                    boards, sequences, lengths, labels = boards.to(device, non_blocking=True), sequences.to(device, non_blocking=True), lengths, labels.to(device, non_blocking=True)
                    outputs = model(boards, sequences, lengths)
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
        train_error.append(100-100*train_correct/train_total)
        print(f'Epoch {epoch+1}, Training Loss: {training_loss/len(train_loader)}, Validation Error: {val_error[-1]}, Validation Top-3 Accuracy: {val_3_accuracy[-1]}, Training Error: {train_error[-1]}')
        for op_params in optimizer.param_groups:
            op_params['lr'] = op_params['lr'] * learn_decay
        torch.save(model.state_dict(), f'model_images/multimodalmodel-exp-{experiment_name}-checkpoint-{epoch}.pth')
    return train_error,train_loss_values, val_error, val_loss_values

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=1, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha  # alpha can be set to a constant, or it can be a tensor of shape (num_classes,)
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # Prevents nans when probability 0
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss
        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('../data/combined/vocab.pkl', 'rb') as inp:
    vocab = pickle.load(inp)
    
fens = pd.read_csv('../data/1500/black/fens.csv')['fens']

# For trainX
dtype_trainX = np.bool_  # or the correct dtype for your data
shape_trainX = (18568313, 12,8,8)  # replace with the correct shape
trainX = load_memmap('./../data/1500/black/trainX_boards.memmap', dtype_trainX, shape_trainX)

# For trainY
dtype_trainY = np.int64 # or the correct dtype for your data
shape_trainY = (18568313,)  # replace with the correct shape
trainY = load_memmap('./../data/1500/black/trainY.memmap', dtype_trainY, shape_trainY)

# For trainY
dtype_trainX_seqlengths = np.int64 # or the correct dtype for your data
shape_trainX_seqlengths = (18568313,)  # replace with the correct shape
trainX_seqlengths = load_memmap('./../data/1500/black/trainX_seqlengths.memmap', dtype_trainX_seqlengths, shape_trainX_seqlengths)

# For trainY
dtype_trainX_sequences = np.int64 # or the correct dtype for your data
shape_trainX_sequences = (18568313, 16)  # replace with the correct shape
trainX_sequences = load_memmap('./../data/1500/black/trainX_sequences.memmap', dtype_trainX_sequences, shape_trainX_sequences)

dataset = MultimodalDatasetWithFEN(trainX_sequences, trainX, trainX_seqlengths, fens, trainY)
# Calculate split sizes
total_size = len(dataset)

# We're scaling the model size so let's bring in more data as well
train_size = int(0.99995 * total_size)
val_size = int(total_size * 0.00004)

# Create subsets for training and validation
train_dataset = Subset(dataset, range(0, train_size))
val_dataset = Subset(dataset, range(train_size, train_size + val_size))
        
# Reload the data with particular batch size
torch.multiprocessing.set_start_method('fork', force=True)
batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=6,pin_memory=True)

# Initialize model, loss function, and optimizer
d_hidden = 256
d_embed = 64
NUM_EPOCHS = 4
d_out = len(vocab.id_to_move.keys())
model = MultiModalSeven(vocab,d_embed,d_hidden,d_out) 
model = model.to(device)
criterion = FocalLoss(gamma=2, alpha=1, reduction='mean')
lr = 1e-3
weight_decay=1e-8
learn_decay = 0.5 # 
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

model.load_state_dict(torch.load('model_images/multimodalmodel-exp-12-white-1500-checkpoint-9.pth'))

print(count_parameters(model))

model.compile()

train_error,train_loss_values, val_error, val_loss_value = train_with_fen(device, 
                                                                          model, 
                                                                          train_loader, 
                                                                          val_loader, 
                                                                          criterion, 
                                                                          optimizer, 
                                                                          NUM_EPOCHS, 
                                                                          learn_decay,
                                                                          '12-white-1500-second')


model.eval()
val_correct_3 = 0
val_correct = 0
val_total = 0

if val_loader is not None:
    with torch.no_grad():
        for boards, sequences, lengths, fens, labels in val_loader:
            boards, sequences, lengths, labels = boards.to(device, non_blocking=True), sequences.to(device, non_blocking=True), lengths, labels.to(device, non_blocking=True)
            outputs = model(boards, sequences, lengths)
            probabilities = torch.softmax(outputs, dim=1)
            minus = 0
            for idx, (sequence, fen, label) in enumerate(zip(sequences, fens, labels)):
                output = probabilities[idx]
                sorted_probs, sorted_indices = torch.sort(output, descending=True)
                chess_board = chess.Board(fen)
                legal_moves_found = 0
                correct_move_found_within_top_3 = False

                for move_idx in sorted_indices:
                    move = vocab.get_move(move_idx.item())  # Convert index to move
                    if is_legal_move(chess_board, move):
                        if legal_moves_found == 0:
                            pred_move = vocab.get_id(move)
                        if vocab.get_id(move) == label.item():  # Check if this legal move is the correct one
                            correct_move_found_within_top_3 = True
                            break
                        legal_moves_found += 1
                        if legal_moves_found == 3:  # Stop after finding top 3 legal moves
                            break
                if pred_move == label.item():
                    val_correct +=1
                if correct_move_found_within_top_3:
                    val_correct_3 += 1
            val_total += (labels.size(0) - minus)

        val_accuracy = 100 * val_correct_3 / val_total
        print(f"Top-3 Validation Accuracy (with only legal moves allowed): {val_accuracy}%")
        val_accuracy = 100 * val_correct / val_total
        print(f"Top-1 Validation Accuracy (with only legal moves allowed): {val_accuracy}%")