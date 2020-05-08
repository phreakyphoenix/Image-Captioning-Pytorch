 
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        #batch normalization
        self.batch= nn.BatchNorm1d(embed_size,momentum = 0.01)
        #Weights initialization
        self.embed.weight.data.normal_(0., 0.02)
        self.embed.bias.data.fill_(0)
        
    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.batch(self.embed(features))
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embed_size= embed_size
        self.drop_prob= 0.2
        self.vocabulary_size = vocab_size
        #Define LSTSM
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size , self.num_layers,batch_first=True)
        self.dropout = nn.Dropout(self.drop_prob)
        self.embed = nn.Embedding(self.vocabulary_size, self.embed_size)
        self.linear = nn.Linear(hidden_size, self.vocabulary_size)
         #Weight initialization
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
    
    def forward(self, features, captions):
        #generating embedings from captures labels
        embeddings = self.embed(captions)
        #Concatenate captions embedidings and images features in one dimension array
        features = features.unsqueeze(1)
        embeddings = torch.cat((features, embeddings[:, :-1,:]), dim=1)

        #Pack in sequences to create several batches with sequence length vocabulary size
        #packed = torch.nn.utils.rnn.pack_padded_sequence(embeddings, self.vocabulary_size,batch_first= True) 
        #LSTM return hidden states and output of LSTM layers (score telling how near we are from finding the right word sequence)
        hiddens, c = self.lstm(embeddings)

        #Regression that feed to the next LSTM cell and contains the previous state
        outputs = self.linear(hiddens)
        #outputs = F.softmax(outputs)

        return outputs
        

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        sampled_ids = []
        #inputs = inputs.unsqueeze(1)
        for i in range(max_len):
            #LSTM cell h, c
            hidden, states = self.lstm(inputs,states)
            outputs = self.linear(hidden.squeeze(1)) 
            #arg max probability per output in LSTM cell 
            _, predicted = outputs.max(1)    
            sampled_ids.append(predicted)
            #Update Hidden state with new output to next LSTM cell
            #How to tell if the index is word-vector index?
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1) 
            
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        sampled_ids = list(sampled_ids.cpu().numpy()[0])
        sampled_ids = [int(i) for i in sampled_ids]
        return  sampled_ids
   
            
            
