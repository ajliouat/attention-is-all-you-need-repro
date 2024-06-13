import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, get_linear_schedule_with_warmup

class TransformerDataset(Dataset):
    def __init__(self, src, tgt):
        self.src = src
        self.tgt = tgt
        
    def __len__(self):
        return len(self.src)
    
    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        src_embed = self.positional_encoding(self.encoder_embedding(src))
        tgt_embed = self.positional_encoding(self.decoder_embedding(tgt))
        memory = self.transformer_encoder(src_embed, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        outs = self.transformer_decoder(tgt_embed, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        return self.fc(outs)

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt, pad_idx):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len).to(device)
    src_mask = torch.zeros((src_seq_len, src_seq_len)).type(torch.bool).to(device)

    src_padding_mask = (src == pad_idx).transpose(0, 1)
    tgt_padding_mask = (tgt == pad_idx).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

# Training loop
def train(model, dataloader, optimizer, scheduler, criterion, device, epochs):
    model.train()
    for epoch in range(epochs):
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt, pad_idx)

            optimizer.zero_grad()
            output = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask, src_key_padding_mask=src_padding_mask, tgt_key_padding_mask=tgt_padding_mask)
            loss = criterion(output.reshape(-1, output.shape[-1]), tgt.reshape(-1))
            loss.backward()
            optimizer.step()
            scheduler.step()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Set up the model, dataset, and training parameters
src_vocab_size = len(src_vocab)
tgt_vocab_size = len(tgt_vocab)
d_model = 512
nhead = 8
num_encoder_layers = 6
num_decoder_layers = 6
dim_feedforward = 2048
dropout = 0.1
batch_size = 32
epochs = 10
lr = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerModel(src_vocab_size, tgt_vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout).to(device)
dataset = TransformerDataset(src_data, tgt_data)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

pad_idx = src_vocab['<pad>'] # or tgt_vocab['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
num_training_steps = len(dataloader) * epochs
num_warmup_steps = num_training_steps // 10
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

train(model, dataloader, optimizer, scheduler, criterion, device, epochs)
