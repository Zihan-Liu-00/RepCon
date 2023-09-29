import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv,GATConv,SAGEConv
from dgl.nn import GlobalAttentionPooling
from dgl.nn import MaxPooling

device = 'cuda' if torch.cuda.is_available() else 'cpu'

############# (a) Sequential Encoder & Predictor #############

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # 10000^{2i/d_model}
        pe[:, 0::2] = torch.sin(position * div_term)  
        pe[:, 1::2] = torch.cos(position * div_term) 
        pe = pe.unsqueeze(0).transpose(0, 1) 
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :] 
        return self.dropout(x)

def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_q, len_k)


def get_attn_subsequence_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask,args):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(args.d_k) 
        scores.masked_fill_(attn_mask, -1e9) 
        
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self,args):
        super(MultiHeadAttention, self).__init__()
        self.args = args
        self.W_Q = nn.Linear(args.d_model, args.d_k * args.n_heads, bias=False)
        self.W_K = nn.Linear(args.d_model, args.d_k * args.n_heads, bias=False)
        self.W_V = nn.Linear(args.d_model, args.d_v * args.n_heads, bias=False)
        self.fc = nn.Linear(args.n_heads * args.d_v, args.d_model, bias=False)

    def forward(self,input_Q, input_K, input_V, attn_mask):
        residual, batch_size = input_Q, input_Q.size(0)

        Q = self.W_Q(input_Q).view(batch_size,-1, self.args.n_heads, self.args.d_k).transpose(1,2) # Q:[batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size,-1, self.args.n_heads, self.args.d_k).transpose(1,2) # K:[batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size,-1, self.args.n_heads, self.args.d_v).transpose(1,2) # V:[batch_size, n_heads, len_v(=len_k, d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.args.n_heads, 1, 1)
        context, attn = ScaledDotProductAttention()(Q,K,V, attn_mask,self.args)
        context = context.transpose(1,2).reshape(batch_size, -1, self.args.n_heads * self.args.d_v)
        output = self.fc(context)

        return nn.LayerNorm(self.args.d_model).to(device)(output+residual),attn 

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self,args):
        super(PoswiseFeedForwardNet, self).__init__()
        self.args = args
        self.fc = nn.Sequential(
            nn.Linear(args.d_model,args.d_ff,bias=False),
            nn.ReLU(),
            nn.Linear(args.d_ff, args.d_model, bias=False)
        )

    def forward(self, inputs):
        '''
        :param inputs: [batch_size, seq_len, d_model]
        :return:
        '''
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(self.args.d_model).to(device)(output+residual) 

class EncoderLayer(nn.Module):
    def __init__(self,args):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(args)
        self.pos_ffn = PoswiseFeedForwardNet(args)

    def forward(self,enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs,enc_inputs,enc_inputs,enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs) 
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self,args):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(args.src_vocab_size_seq, args.d_model)
        self.pos_emb = PositionalEncoding(args.d_model)
        self.layers = nn.ModuleList([EncoderLayer(args) for _ in range(args.n_layers)])

    def forward(self, enc_inputs):
        enc_outputs = self.src_emb(enc_inputs)
        enc_outputs = self.pos_emb(enc_outputs.transpose(0,1)).transpose(0,1) 
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs,enc_inputs) 
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

class Transformer(nn.Module):
    def __init__(self,args):
        super(Transformer,self).__init__()
        self.args = args
        self.encoder = Encoder(args).to(device)
        self.fc = nn.Linear(args.src_len*args.d_model, args.hidden).to(device)
        self.layers = nn.ModuleList([nn.Linear(args.hidden, args.hidden).to(device) for _ in range(args.fc_layers)])
        self.predictor = nn.Linear(args.hidden, args.output_layer).to(device)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self,enc_inputs):
        '''
        :param enc_inputs: [batch_size, src_len]
        :param dec_inputs: [batch_size, tgt_len]
        :return:
        '''
        # Sequential Encoder
        enc_outputs,_ = self.encoder(enc_inputs)
        # Reshape the token-level repr. to sequential repr. h_seq
        dec_inputs = torch.reshape(enc_outputs,(enc_outputs.shape[0],-1))
        hidden = self.fc(dec_inputs)
        h = hidden.clone()
        # Predictor
        for layer in self.layers:
            h = layer(F.leaky_relu(h))
        y = self.predictor(F.leaky_relu(h))

        if self.args.task_type == 'Classification':
            return F.log_softmax(y, dim=1), hidden
        elif self.args.task_type == 'Regression':
            return y, hidden

############# (a) Sequential Encoder & Predictor #############

class GNNs(nn.Module):
    def __init__(self, args):
        super(GNNs, self).__init__()

        self.args = args
        self.convlayers = nn.ModuleList([SAGEConv(args.d_graph, args.d_graph,args.GraphSAGE_aggregator)  for _ in range(args.conv_layers)])
        self.predictor = nn.Linear(args.hidden, args.output_layer)

        hidden_layer = [args.hidden,args.hidden,args.hidden,args.hidden]
        self.e1 = nn.Linear(args.d_graph, hidden_layer[0])
        self.e2 = nn.Linear(hidden_layer[0], hidden_layer[1])
        self.e3 = nn.Linear(hidden_layer[1], hidden_layer[2])
        self.e4 = nn.Linear(hidden_layer[2], hidden_layer[3])

        self.dropout = nn.Dropout(p=0.3)
        self.src_emb = nn.Embedding(args.src_vocab_size_graph, args.d_graph)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, g):

        # Graphical Encoder
        with g.local_scope():
            h = self.src_emb(g.ndata['feat'])
            for layer in self.convlayers:
                h = F.relu(layer(g, h))
            
            g.ndata['h'] = h    
        # Readout with mean pooling
            hg = dgl.mean_nodes(g,'h')

        hidden = F.relu(self.e1(hg))
        # Predictor
        h_2 = F.relu(self.e2(hidden))
        h_3 = F.relu(self.e3(h_2))
        h_4 = F.relu(self.e4(h_3))
        y = self.predictor(h_4)
        
        if self.args.task_type == 'Classification':
            return F.log_softmax(y, dim=1),hidden
        elif self.args.task_type == 'Regression':
            return y,hidden
