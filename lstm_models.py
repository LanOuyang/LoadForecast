import torch
from torch.cuda import _device
import torchvision
import tensorboard
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim


class LSTM(nn.Module):
    def __init__(self, 
            input_size, 
            hidden_size, 
            embedding_size,
            output_size=1, 
            num_layer=2, 
            dropout=0.2, 
            **kwargs):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.input_embedding_layer = nn.Linear(self.input_size, self.embedding_size)
        

        self.lstm = nn.LSTM(input_size, hidden_size,num_layers=num_layer,dropout=dropout)
        self.linear = nn.Linear(hidden_size, output_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden=None):
        batch_size = x.shape[0]
        # if hidden == None:
        #     self.hidden = (torch.zeros(1, batch_size, self.hidden_size),
        #                    torch.zeros(1, batch_size, self.hidden_size))
        # else:
        #     self.hidden = hidden

        # Input shape: (sequence length, batch sizem feature size)
        x = x.permute((1, 0, 2))
        #if x.shape[1] == 1:
        #    x = x.squeeze(1)

        # encode raw features before feeding into lstm
        input_embedded = self.dropout(self.relu(self.input_embedding_layer(x)))
        

        lstm_out, (self.hidden_states, self.cell_states) = self.lstm(x, hidden)
        predictions = self.linear(lstm_out)

        output = predictions[-1].view(batch_size, -1, self.output_size)

        return output, (self.hidden_states.detach(), self.cell_states.detach())



class SocialLSTM(nn.Module):
    def __init__(self,
            device,
            input_size, 
            hidden_size, 
            embedding_size,
            output_size = 1, 
            app_num = 21,
            num_layer = 2, 
            dropout = 0.2, 
            **kwargs):
        """
        input_size includes app_size
        appliances' data individually fed into lstm;
        sum hidden state or concatenate?;
        fed into lstm
        OR
        appliances'data fed into lstm collectively

        combine with other features (how is this operation in back prop?);
        """
        super(SocialLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.input_size = input_size
        self.output_size = output_size
        self.num_layer = num_layer
        self.app_num = app_num
        self.device = device
        # self.seq_len = seq_len

        self.input_embedding_layer = nn.Linear(self.input_size, self.embedding_size)
        self.tensor_embedding_layer = nn.Linear(self.app_num*self.hidden_size, self.embedding_size)

        self.cell = nn.LSTM(2*self.embedding_size, self.hidden_size,num_layers=self.num_layer,dropout=dropout)
        self.output_layer = nn.Linear(self.app_num*self.hidden_size, self.output_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def getSocialTensor(self, hidden_states):
        socialTensor = hidden_states.view(1, 1, self.app_num*self.hidden_size)
        socialTensor = socialTensor.expand(-1, self.app_num, -1)

        return socialTensor

    def forward(self, x, hidden_states=None):
        # app indices in input: app_range=[1,2]+list(range(4,23))

        # print("input data size:",x.shape)

        # batch size should be 1
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        feature_num = x.shape[2]

        
        if hidden_states == None:
          hidden_states = torch.zeros(self.num_layer, self.app_num, self.hidden_size)  
          cell_states = torch.zeros(self.num_layer, self.app_num, self.hidden_size)
        else:
          (hidden_states, cell_states) = hidden_states
        hidden_states = hidden_states.to(self.device)
        cell_states = cell_states.to(self.device)
        

        x = x.permute((1, 0, 2))
        # x = x.squeeze(0)

        # reshape input data 
        # features = torch.cat((x[:, :, 0:1], x[:, :, 3:4],x[:, :, 23:]), axis=2).expand(-1, self.app_num, -1)
        # app_data = torch.cat((x[:, :, 1:3], x[:, :, 4:23]), axis=2).view(-1, self.app_num, 1)
        features = x[:, :, 0:-21].expand(-1, self.app_num, -1)
        app_data = x[:, :, -21:].view(-1, self.app_num, 1)
        input_data = torch.cat((features, app_data), axis=2)
        
        # Again, batch size should be 1
        outputs = torch.zeros(batch_size, seq_len, self.output_size).to(self.device)

        for i in range(seq_len):
            # all appliances in a batch

            input_current = input_data[i:i+1]
            
            # aggregate hidden states of all appliances
            social_tensor = self.getSocialTensor(hidden_states[-1])
            #print("hidden states device:", hidden_states.device)
            #print("social tensor device:", social_tensor.device)

            # embed other feature besides appliances electricity usage
            input_embedded = self.dropout(self.relu(self.input_embedding_layer(input_current)))
            # embed social tensor
            tensor_embedded = self.dropout(self.relu(self.tensor_embedding_layer(social_tensor)))

            #print("input_embedding shape: ", input_embedded.shape)
            #print("tensor_embedding shape: ", tensor_embedded.shape)
            concat_embedded = torch.cat((input_embedded, tensor_embedded), 2)
            
            cell_output, (hidden_states, cell_states) = self.cell(concat_embedded, (hidden_states, cell_states))

            # one-layer lstm and one-layer (the last layer) hidden states
            # be careful when using multi-layer lstm
            outputs[0,i] = self.output_layer(hidden_states[-1].flatten())

        # reshape outputs
        # outputs = outputs.squeeze(2)

        # outputs = self.output_layer(hidden_states[-1].flatten()).view(batch_size, -1, self.output_size)

        return outputs, (hidden_states.detach(), cell_states.detach())


