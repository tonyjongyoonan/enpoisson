import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_CHANNELS = 12

class ChessCNN(nn.Module):
    def __init__(self, d_out):
        super(ChessCNN, self).__init__()
        # Assuming each channel represents a different piece type (e.g., 6 channels for 6 types each)
        self.conv1 = nn.Conv2d(INPUT_CHANNELS, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)  # Batch normalization for first conv layer
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # Batch normalization for second conv layer
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)  # Batch normalization for second conv layer
        self.fc1 = nn.Linear(64 * 8 * 8, 64)  # Assuming an 8x8 chess board
        self.fc2 = nn.Linear(64, d_out)

    def forward(self, x):
        # Apply first convolution, followed by batch norm, then ReLU
        x = F.relu(self.bn1(self.conv1(x)))
        # Apply second convolution, followed by batch norm, then ReLU
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # Flatten the tensor
        x = x.view(x.size(0), -1)

        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class ChessCNN_no_pooling(nn.Module):
    def __init__(self, d_out):
        super(ChessCNN_no_pooling, self).__init__()
        # Assuming each channel represents a different piece type (e.g., 6 channels for 6 types each)
        self.conv1 = nn.Conv2d(INPUT_CHANNELS, 36, kernel_size=3, padding=1)
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
        self.conv1 = nn.Conv2d(INPUT_CHANNELS, 64, kernel_size=3, padding=1)
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
            INPUT_CHANNELS, 64, kernel_size=3, stride=1, padding=1, bias=False
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

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.se = SELayer(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.se(x)
        return x

class ConvBlockTwo(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlockTwo, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.conv2 = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.se = SELayer(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        # x = F.relu(x)
        # x = self.conv2(x)
        # x = self.bn(x)
        x = self.se(x)
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

class SENet(nn.Module):
    def __init__(self, d_out):
        super(SENet, self).__init__()
        self.conv1 = ConvBlock(INPUT_CHANNELS, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvBlock(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = ConvBlock(64, 64, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(64 * 8 * 8, d_out)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class SENet_Channel_Wise(nn.Module):
    def __init__(self, d_out):
        super(SENet_Channel_Wise, self).__init__()
        self.conv1 = ConvBlock(INPUT_CHANNELS, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvBlock(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = ConvBlock(64, 64, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x.view(x.size(0), x.size(1), -1)
    
class SENetTwo(nn.Module):
    def __init__(self, d_out):
        super(SENetTwo, self).__init__()
        self.conv0 = nn.Conv2d(INPUT_CHANNELS, 64, kernel_size = 3, stride = 1, padding = 1)
        self.conv1 = ConvBlockTwo(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvBlockTwo(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = ConvBlockTwo(64, 64, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(64 * 8 * 8, d_out)

    def forward(self, x):
        x = self.conv0(x)
        
        # First ConvBlockTwo
        identity = x  # Save input for residual connection
        out = self.conv1(x)
        out += identity  # Add input (residual connection)
        out = F.relu(out)  # Apply ReLU activation

        # Second ConvBlockTwo
        identity = out  # Update identity to output of previous block
        out = self.conv2(out)
        out += identity  # Add input (residual connection)
        out = F.relu(out)  # Apply ReLU activation

        # Third ConvBlockTwo
        identity = out  # Update identity to output of previous block
        out = self.conv3(out)
        out += identity  # Add input (residual connection)
        out = F.relu(out)  # Apply ReLU activation

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class SENetThree(nn.Module):
    def __init__(self, d_out):
        super(SENetThree, self).__init__()
        self.conv1 = ConvBlock(INPUT_CHANNELS, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvBlock(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = ConvBlock(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = ConvBlock(64, 64, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(64 * 8 * 8, d_out)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
class RNNModel(nn.Module):
    def __init__(
        self,
        vocab,
        d_embed,
        d_hidden,
        d_out,
        dropout=0.5,
        num_layers=2,
        bidirectional=False,
        embedding_matrix=None,
    ):
        super(RNNModel, self).__init__()
        self.embeddings = nn.Embedding(len(vocab.move_to_id), d_embed)
        # self.embeddings = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.lstm = nn.LSTM(
            d_embed,
            d_hidden,
            dropout=dropout,
            bidirectional=bidirectional,
            num_layers=num_layers,
        )
        self.fc = nn.Sequential(nn.Linear(d_hidden, d_out))

    def forward(self, x, seq_lengths):
        x = self.embeddings(x)
        # Sort x and seq_lengths in descending order
        # This is required for packing the sequence
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        x = x[perm_idx]
        # Pack the sequence
        packed_input = pack_padded_sequence(x, seq_lengths, batch_first=True)
        # Pass the packed sequence through the LSTM
        packed_output, (hidden, cell) = self.lstm(packed_input)

        # Unpack the sequence
        output, _ = pad_packed_sequence(
            packed_output, batch_first=True, total_length=x.size()[1]
        )
        _, unperm_idx = perm_idx.sort(0)
        unperm_idx = unperm_idx.to(device, non_blocking=True)
        output = output.index_select(0, unperm_idx)
        # This takes all the outputs across the cells
        mean_pooled = torch.mean(output, dim=1)
        # output = torch.cat((mean_pooled,hidden[-1]),dim=1)
        output = self.fc(mean_pooled)
        return output

class RNNModel_Token_Wise(nn.Module):
    def __init__(
        self,
        vocab,
        d_embed,
        d_hidden,
        dropout=0.5,
        num_layers=2,
        bidirectional=False,
        embedding_matrix=None,
    ):
        super(RNNModel_Token_Wise, self).__init__()
        self.embeddings = nn.Embedding(len(vocab.move_to_id), d_embed)
        # self.embeddings = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.lstm = nn.LSTM(
            d_embed,
            d_hidden,
            dropout=dropout,
            bidirectional=bidirectional,
            num_layers=num_layers,
        )

    def forward(self, x, seq_lengths):
        x = self.embeddings(x)
        # Sort x and seq_lengths in descending order
        # This is required for packing the sequence
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        x = x[perm_idx]
        # Pack the sequence
        packed_input = pack_padded_sequence(x, seq_lengths, batch_first=True)
        # Pass the packed sequence through the LSTM
        packed_output, (hidden, cell) = self.lstm(packed_input)

        # Unpack the sequence
        output, _ = pad_packed_sequence(
            packed_output, batch_first=True, total_length=x.size()[1]
        )
        _, unperm_idx = perm_idx.sort(0)
        unperm_idx = unperm_idx.to(device, non_blocking=True)
        output = output.index_select(0, unperm_idx)
        # This takes all the outputs across the cells
        return output
    
class RNNModelTwo(nn.Module):
    def __init__(
        self,
        vocab,
        d_embed,
        d_hidden,
        d_out,
        dropout=0.5,
        num_layers=2,
        bidirectional=False,
        embedding_matrix=None,
    ):
        super(RNNModelTwo, self).__init__()
        self.embeddings = nn.Embedding(len(vocab.move_to_id), d_embed)
        # self.embeddings = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.lstm = nn.LSTM(
            d_embed,
            d_hidden,
            dropout=dropout,
            bidirectional=bidirectional,
            num_layers=num_layers,
        )
        self.fc = nn.Sequential(nn.Linear(16*d_hidden, d_out))
        if bidirectional:
            self.fc = nn.Sequential(nn.Linear(2*16*d_hidden, d_out))

    def forward(self, x, seq_lengths):
        x = self.embeddings(x)
        # Sort x and seq_lengths in descending order
        # This is required for packing the sequence
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        x = x[perm_idx]
        # Pack the sequence
        packed_input = pack_padded_sequence(x, seq_lengths, batch_first=True)
        # Pass the packed sequence through the LSTM
        packed_output, (hidden, cell) = self.lstm(packed_input)

        # Unpack the sequence
        output, _ = pad_packed_sequence(
            packed_output, batch_first=True, total_length=x.size()[1]
        )
        _, unperm_idx = perm_idx.sort(0)
        unperm_idx = unperm_idx.to(device, non_blocking=True)
        output = output.index_select(0, unperm_idx)
        # This takes all the outputs across the cells
        output = output.view(output.size(0), -1) 
        output = self.fc(output)
        return output
    
class TransformerModel(nn.Module):
    def __init__(self, vocab, d_embed, nhead, d_hidden, d_out, num_layers, dropout=0.2):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab.move_to_id), d_embed)
        self.pos_encoder = PositionalEncoding(d_embed, dropout)
        transformer_layers = nn.TransformerEncoderLayer(d_embed, nhead, d_hidden, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layers, num_layers)
        self.d_embed = d_embed

    def forward(self, x, seq_lengths):
        seq_lengths = seq_lengths.to(device)
        mask = self.create_mask(seq_lengths, x.size(1))
        x = self.embedding(x) * math.sqrt(self.d_embed)
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x, src_key_padding_mask=mask)
        output = output.mean(dim=1)
        #last_token_output = output[:, -1, :]  # shape (batch_size, d_embed)
        return output
    
    def create_mask(self, seq_lengths, max_len):
        batch_size = seq_lengths.size(0)
        # Create a mask of shape (batch_size, max_len) with all zeros
        mask = torch.zeros(batch_size, max_len, device=seq_lengths.device)

        # Iterate over each element in seq_lengths to set `-inf` for padding
        for i in range(batch_size):
            length = seq_lengths[i]
            mask[i, length:] = float('-inf')

        return mask

class PositionalEncoding(nn.Module):
    def __init__(self, d_embed, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_embed)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_embed, 2).float() * (-math.log(10000.0) / d_embed))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)
    
class MultiModal(nn.Module):
    def __init__(self, vocab, d_embed, d_hidden, d_out, dropout=0.5) -> None:
        super().__init__()
        self.rnn = RNNModel(vocab, d_embed, d_hidden, 16, dropout=dropout)
        self.cnn = ChessCNN(32)
        self.fc = nn.Linear(48, d_out)

    def forward(self, board, sequence, seq_lengths):
        seq_encoding = self.rnn(sequence, seq_lengths)
        cnn_encoding = self.cnn(board)
        pred = self.fc(torch.cat((seq_encoding, cnn_encoding), dim=1))
        return pred
    
class MultiModalTwo(nn.Module):
    def __init__(self, vocab, d_embed, d_hidden, d_out, dropout=0.5) -> None:
        super().__init__()
        self.rnn = RNNModel(vocab, d_embed, d_hidden, 16, dropout=dropout)
        self.cnn = SENet(64)
        self.fc = nn.Sequential(nn.Linear(16 + 64, 64), nn.ReLU(), nn.Linear(64, d_out))

    def forward(self, board, sequence, seq_lengths):
        seq_encoding = self.rnn(sequence, seq_lengths)
        cnn_encoding = self.cnn(board)
        pred = self.fc(torch.cat((seq_encoding, cnn_encoding), dim=1))
        return pred



class MultiModalThree(nn.Module):
    def __init__(self, vocab, d_embed, d_encoding, d_out, dropout=0.5) -> None:
        super().__init__()
        # rnn needs to output 64 dimensions per token so it can interact with the 8x8 board
        # channels is kind of like "heads" here
        self.rnn = RNNModel_Token_Wise(vocab, d_embed, 64, dropout=dropout)
        self.cnn = SENet_Channel_Wise(d_encoding)

        # Attention layers
        self.seq_query = nn.Linear(64, 64)
        self.img_keys = nn.Linear(64, 64)
        self.img_values = nn.Linear(64, 64)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(16*64, 64),  # Adjusted to combine attention outputs
            nn.ReLU(),
            nn.Linear(64, d_out),
        )

    def forward(self, board, sequence, seq_lengths):
        # Encode sequence of moves
        seq_encoding = self.rnn(sequence, seq_lengths)  # [batch_size, seq_length, d_dimensions]

        # Encode board state
        cnn_encoding = self.cnn(board)  # [batch_size, channels, d_dimensions]

        Q = self.seq_query(seq_encoding)
        K = self.img_keys(cnn_encoding)
        V = self.img_values(cnn_encoding)

        # Compute attention scores
        # Ensure dimensions are compatible for bmm (e.g., [batch_size, seq_length, d_dimensions] @ [batch_size, d_dimensions, channels])
        attn_scores = torch.bmm(Q, K.transpose(1, 2))  # [batch_size, seq_length, channels]

        # Normalize attention scores
        attn_weights = F.softmax(attn_scores, dim=-1)  # [batch_size, seq_length, channels]

        # Apply attention to values (still using cnn_encoding as values)
        attention_applied = torch.bmm(attn_weights, V.transpose(1, 2))  # [batch_size, seq_length, d_dimensions]

        # Prediction
        attention_applied = attention_applied.view(attention_applied.size(0),-1)
        pred = self.fc(attention_applied) 

        return pred

class MultiModalFour(nn.Module):
    def __init__(self, vocab, d_embed, d_hidden, d_out, dropout=0.5) -> None:
        super().__init__()
        self.rnn = RNNModel(vocab, d_embed, d_hidden, 16, dropout=dropout)
        self.cnn = SENetTwo(64)
        self.fc = nn.Sequential(nn.Linear(16 + 64, 64), nn.ReLU(), nn.Linear(64, d_out))

    def forward(self, board, sequence, seq_lengths):
        seq_encoding = self.rnn(sequence, seq_lengths)
        cnn_encoding = self.cnn(board)
        pred = self.fc(torch.cat((seq_encoding, cnn_encoding), dim=1))
        return pred
    
class MultiModalFive(nn.Module):
    def __init__(self, vocab, d_embed, d_hidden, d_out, dropout=0.5) -> None:
        super().__init__()
        self.rnn = RNNModelTwo(vocab, d_embed, d_hidden, 64, dropout=dropout)
        self.cnn = SENet(128)
        self.fc = nn.Sequential(nn.Linear(64 + 128, 256), nn.ReLU(), nn.Linear(256, d_out))

    def forward(self, board, sequence, seq_lengths):
        seq_encoding = self.rnn(sequence, seq_lengths)
        cnn_encoding = self.cnn(board)
        pred = self.fc(torch.cat((seq_encoding, cnn_encoding), dim=1))
        return pred
    
class MultiModalSix(nn.Module):
    def __init__(self, vocab, d_embed, d_hidden, d_out, dropout=0.5) -> None:
        super().__init__()
        self.rnn = RNNModelTwo(vocab, d_embed, d_hidden, 64, dropout=dropout, bidirectional=True)
        self.cnn = SENet(128)
        self.fc = nn.Sequential(nn.Linear(64 + 128, 256), nn.ReLU(), nn.Linear(256, d_out))

    def forward(self, board, sequence, seq_lengths):
        seq_encoding = self.rnn(sequence, seq_lengths)
        cnn_encoding = self.cnn(board)
        pred = self.fc(torch.cat((seq_encoding, cnn_encoding), dim=1))
        return pred
    
class MultiModalSeven(nn.Module):
    def __init__(self, vocab, d_embed, d_hidden, d_out, dropout=0.5) -> None:
        super().__init__()
        self.rnn = RNNModelTwo(vocab, d_embed, d_hidden, 128, dropout=dropout, bidirectional=True)
        self.cnn = SENetThree(256)
        self.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(256+128, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, d_out))

    def forward(self, board, sequence, seq_lengths):
        seq_encoding = self.rnn(sequence, seq_lengths)
        cnn_encoding = self.cnn(board)
        pred = self.fc(torch.cat((seq_encoding, cnn_encoding), dim=1))
        return pred
    
class MultiModalEight(nn.Module):
    def __init__(self, vocab, d_embed, d_hidden, d_out, dropout=0.5) -> None:
        super().__init__()
        nhead = 12
        self.transform = TransformerModel(vocab, d_embed, nhead, d_hidden, d_out, num_layers=3,dropout=0.1) 
        self.cnn = SENetThree(256)
        self.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(256+d_embed, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, d_out))

    def forward(self, board, sequence, seq_lengths):
        seq_encoding = self.transform(sequence, seq_lengths)
        cnn_encoding = self.cnn(board)
        pred = self.fc(torch.cat((seq_encoding, cnn_encoding), dim=1))
        return pred
    
class CNNtoTransformer(nn.Module):
    def __init__(self, vocab, d_embed, d_hidden, d_out, dropout=0.5) -> None:
        super().__init__()
        self.cnn = SENetThree(256)
        self.transformer
        self.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, d_out))

    def forward(self, board, sequence, seq_lengths):
        cnn_encoding = self.cnn(board)
        pred = self.fc(torch.cat((seq_encoding, cnn_encoding), dim=1))
        return pred