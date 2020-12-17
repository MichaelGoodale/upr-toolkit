import time
import random
import math

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
from tqdm import tqdm

from upr_toolkit.models import ModelData
from upr_toolkit.analyses import get_sentence_tensor

class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.rnn = nn.LSTM(input_dim, hid_dim, n_layers, dropout = dropout)
        
    def forward(self, src):
        '''Takes a source utterance with shape (batch_size, utterance_length, input_dim)'''
        outputs, (hidden, cell) = self.rnn(src.transpose(0,1))
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.rnn = nn.LSTM(output_dim, hid_dim, n_layers, dropout = dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)

    def forward(self, src, hidden, cell):
        #input = [batch size]
        src = nn.functional.one_hot(src, num_classes=self.output_dim).unsqueeze(0).float()
        output, (hidden, cell) = self.rnn(src, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        #src = [src len, batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)
        
        #first input to the decoder is the <sos> tokens
        input = trg[:, 0]
        
        for t in range(1, trg_len):
            
            #insert input token embedding, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)
            
            #place predictions in a tensor holding predictions for each token
            outputs[:, t, :] = output
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            #get the highest predicted token from our predictions
            top1 = output.argmax(1) 
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[:, t] if teacher_force else top1
        
        return outputs

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def train(model, iterator, optimizer, criterion, clip, epoch_num, total_epochs, device):
    
    model.train()
    
    epoch_loss = 0
    
    for i, (src, trg) in tqdm(enumerate(iterator), desc=f"Training epoch {epoch_num+1}/{total_epochs}", total=len(iterator.dataset)):
        
        optimizer.zero_grad()
        src = src.to(device)
        trg = trg.to(device)
        output = model(src, trg)
        output_dim = output.shape[-1]

        output = output[:, 1:, :].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)

        
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        print(loss.item())
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, (src, trg) in enumerate(iterator):
            src = src.to(device)
            trg = trg.to(device)
            output = model(src, trg, 0) #turn off teacher forcing
            output_dim = output.shape[-1]
            output = output[:, 1:, :].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)


def train_model_on_prosody(batch_size=16, hidden_dim=128, n_layers=3, n_epochs=10, generate_data=True):
    params = {'batch_size': 16,
              'shuffle': True,
              'num_workers': 6}

    y_map = {"0":0, "1":1, "2":2, "-": 3, "<pad>": 4, "<sos>": 5}

    if generate: 
        data = ModelData(cache_file="/home/michael/Documents/Cogmaster/M1/S1/stage/model_caches/cpc_eng_data.ft")
        X, y = get_sentence_tensor(data.train)

        y_pad = max(len(s) for s in y)

        y = torch.tensor([[y_map["<sos>"]]+[y_map[s] for s in sentence]+[4]*(y_pad-len(sentence)) for sentence in y])

        test_X, test_y = get_sentence_tensor(data.test)
        test_y = torch.tensor([[y_map["<sos>"]]+[y_map[s] for s in sentence]+[4]*(y_pad-len(sentence)) for sentence in test_y])
        torch.save(X, "X.pt")
        torch.save(y, "y.pt")
        torch.save(test_X, "test_X.pt")
        torch.save(test_y, "test_y.pt")
    else:
        X = torch.load("X.pt")
        y = torch.load("y.pt")
        test_X = torch.load("test_X.pt")
        test_y = torch.load("test_y.pt")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = X.shape[-1]

    val_idx = int(0.95*X.shape[1])
    training_dataset = TensorDataset(X[:val_idx, :, :], y[:val_idx])
    training_generator = DataLoader(training_dataset, **params)
    val_dataset = TensorDataset(X[val_idx:, :, :], y[val_idx:])
    val_generator = DataLoader(val_dataset, **params)
    test_dataset = TensorDataset(test_X, test_y)
    test_generator = DataLoader(test_dataset, **params)


    OUTPUT_DIM = len(y_map) 
    DROPOUT = 0.1

    enc = Encoder(input_dim, hidden_dim, n_layers, DROPOUT)
    dec = Decoder(OUTPUT_DIM, hidden_dim, n_layers, DROPOUT)
    model = Seq2Seq(enc, dec, device).to(device)

            
    model.apply(init_weights)

    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=4)

    CLIP = 1
    best_valid_loss = float('inf')


    for epoch in range(n_epochs):
        
        start_time = time.time()
        
        train_loss = train(model, training_generator, optimizer, criterion, CLIP, epoch, n_epochs, device)
        valid_loss = evaluate(model, val_generator, criterion, device)
        
        end_time = time.time()
        
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut1-model.pt')
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
