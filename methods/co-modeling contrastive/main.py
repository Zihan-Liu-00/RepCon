import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import pandas as pd
from utils_seq import *
from utils_graph import *
from model import *
from loss import info_nce_loss
import argparse
import warnings
import os

parser = argparse.ArgumentParser()

# Model & Training Settings
parser.add_argument('--dataset', type=str, default='PepDB',
                    choices=['AMP','AP','PepDB','RT'])
parser.add_argument('--mode', type=str, default='train',
                    choices=['train','test'])
parser.add_argument('--seed', type=int, default=5, 
                    help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--seq_lr', type=float, default=1e-4,
                    help='Initial learning rate.')
parser.add_argument('--graph_lr', type=float, default=5e-4,
                    help='Initial learning rate.')

# Sequential Model Parameters
parser.add_argument('--src_vocab_size_seq', type=int, default=21,
                    help='20 natural amino acids plus padding token')
parser.add_argument('--model_seq', type=str, default='Transformer')
parser.add_argument('--d_model', type=int, default=64,
                    help='Output dim of self-attention layers')
parser.add_argument('--fc_layers', type=int, default=4,
                    help='Predictor MLP layers')
parser.add_argument('--d_ff', type=int, default=2048, 
                    help='Hidden dim of FFN')
parser.add_argument('--d_k', type=int, default=64, 
                    help='Hidden dim of K and Q')
parser.add_argument('--d_v', type=int, default=64, 
                    help='Hidden dim of V')
parser.add_argument('--n_layers', type=int, default=6, 
                    help='Num of self-attention layers')
parser.add_argument('--n_heads', type=int, default=8, 
                    help='Num of head for multi-head attention')

# Graphical Model Parameters
parser.add_argument('--model_graph', type=str, default='GraphSAGE')
parser.add_argument('--GraphSAGE_aggregator', type=str, default='lstm',
                    help='Aggregation function of GraphSAGE')
parser.add_argument('--conv_layers', type=int, default=2)
parser.add_argument('--d_graph', type=int, default=256)
parser.add_argument('--src_vocab_size_graph', type=int, default=15,
                    help='15 types of beads') 

# InfoNCE Params
parser.add_argument('--n_views', type=int, default=2, 
                    help='Num of positive pairs')
parser.add_argument('--temperature', type=float, default=0.07)
parser.add_argument('--nce_weight', type=float, default=1e-2)

args = parser.parse_args()

# The type of task
if args.dataset in ['AP','RT']:
    args.task_type = 'Regression'
elif args.dataset in ['AMP','PepDB']:
    args.task_type = 'Classification'
else: 
    warnings.warn('Dataset with undefined task')

# The maximum length of the peptides from each dataset 
if args.dataset in ['AP']:
    args.src_len = 10
elif args.dataset == ['RT','AMP','PepDB']:
    args.src_len = 50
else: 
    args.src_len = 100

# Set the default device 
args.device = 'cuda' if torch.cuda.is_available() else 'cpu' 
# Set model saving/loading path
if not os.path.exists('methods/co-modeling contrastive/saved_models'):
    os.mkdir('methods/co-modeling contrastive/saved_models')
args.model_path='methods/co-modeling contrastive/saved_models/'

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)  

def main():

    # Read raw peptide dataset in FASTA form
    df_seq_train = pd.read_csv('seq dataset/{}/train.csv'.format(args.dataset))
    df_seq_valid = pd.read_csv('seq dataset/{}/valid.csv'.format(args.dataset))
    df_seq_test = pd.read_csv('seq dataset/{}/test.csv'.format(args.dataset))

    # Convert dataset from amino acid sequences to molecular graphs
    if not os.path.exists('graph dataset/{}'.format(args.dataset)):
        os.mkdir('graph dataset/{}'.format(args.dataset))
    if not os.path.exists('graph dataset/{}/test.bin'.format(args.dataset)):
        make_dgl_dataset(args)

    # Build data in DGL graph (coarse-grained molecular graphs)
    train_dataset_graph, train_label = dgl.load_graphs('graph dataset/{}/train.bin'.format(args.dataset))
    valid_dataset_graph, valid_label = dgl.load_graphs('graph dataset/{}/valid.bin'.format(args.dataset))
    test_dataset_graph, test_label = dgl.load_graphs('graph dataset/{}/test.bin'.format(args.dataset))
    
    # Process the labels
    if args.task_type == 'Classification':
        train_label = train_label['labels'].long()
        valid_label = valid_label['labels'].long()
        test_label = test_label['labels'].long()
        args.output_layer = int(train_label.max())-int(train_label.min())+1
    elif args.task_type == 'Regression':
        train_label = train_label['labels'].float()
        valid_label = valid_label['labels'].float()
        test_label = test_label['labels'].float()
        args.output_layer = 1
        args.label_max = train_label.max().item()
        args.label_min = train_label.min().item()
        # Normalize the regression labels by min-max
        train_label = (train_label - args.label_min) / (args.label_max-args.label_min)
        valid_label = (valid_label - args.label_min) / (args.label_max-args.label_min)
        test_label = (test_label - args.label_min) / (args.label_max-args.label_min)
    
    # Convert sequential peptide data from pandas dataframe to torch tensor
    train_feat = np.array(df_seq_train["Feature"])
    valid_feat = np.array(df_seq_valid["Feature"])
    test_feat = np.array(df_seq_test["Feature"])
    train_dataset_seq = make_seq_data(train_feat,args.src_len)
    valid_dataset_seq = make_seq_data(valid_feat,args.src_len).to(args.device)
    test_dataset_seq = make_seq_data(test_feat,args.src_len).to(args.device)

    # Build DataLoaders
    train_dataset = MyDataSet(train_dataset_graph,train_dataset_seq,train_label)
    train_loader = Data.DataLoader(train_dataset, args.batch_size, True,
                        collate_fn=collate)
    valid_dataset = MyDataSet(valid_dataset_graph,valid_dataset_seq,valid_label)
    valid_loader = Data.DataLoader(valid_dataset, args.batch_size, False,
                        collate_fn=collate)
    test_dataset = MyDataSet(test_dataset_graph,test_dataset_seq,test_label)
    test_loader = Data.DataLoader(test_dataset, args.batch_size, False,
                        collate_fn=collate)
    
    #---------------- Train Phase ----------------#

    if args.mode == 'train':

        # Initialize the models and the optimizers
        Seq_model = Transformer(args).to(args.device)
        Seq_optimizer = optim.Adam(Seq_model.parameters(), lr=args.seq_lr)
        Graph_model = GNNs(args).to(args.device)
        Graph_optimizer = optim.Adam(Graph_model.parameters(), lr=args.graph_lr)

        # Initialize MSE loss and CE loss for Infonce
        loss_mse = torch.nn.MSELoss().to(args.device)
        loss_infonce = torch.nn.CrossEntropyLoss().to(args.device)

        # Initialize the validation performance
        valid_mse_saved = 1e5
        valid_acc_saved = 0

        # Train models by epoches of training data
        for epoch in range(args.epochs):
            Seq_model.train()
            Graph_model.train()
            # Extract a batch of data
            for graphs,sequences,labels in train_loader:
                graphs = graphs.to(args.device)
                sequences = sequences.to(args.device)
                labels = labels.to(args.device)
                
                # Forward Models
                seq_outputs, seq_hid = Seq_model(sequences)
                graph_outputs, graph_hid = Graph_model(graphs)

                # Supervised loss
                if args.task_type == 'Classification':
                    loss_seq = F.nll_loss(seq_outputs, labels)
                    loss_graph = F.nll_loss(graph_outputs, labels)
                elif args.task_type == 'Regression':
                    loss_seq = loss_mse(seq_outputs, labels)
                    loss_graph = loss_mse(graph_outputs, labels)

                # Unsupervised InfoNCE loss
                hid_pairs = torch.cat([seq_hid, graph_hid], 0)
                logits, cont_labels = info_nce_loss(args,hid_pairs)
                l_infonce = args.nce_weight*loss_infonce(logits, cont_labels)

                # Overall training loss
                loss = loss_seq + loss_graph + l_infonce

                # Update model parameters
                Seq_optimizer.zero_grad()
                Graph_optimizer.zero_grad()
                loss.backward()
                Seq_optimizer.step()
                Graph_optimizer.step()

            # Validation
            if (epoch+1) % 1 == 0:
                with torch.no_grad():
                    Seq_model.eval()
                    Graph_model.eval()

                    # Derive prediction results for all samples in validation set
                    predicts = []
                    gra_predicts = []
                    for graphs,sequences,labels in valid_loader:
                        outputs,_ = Seq_model(sequences)
                        gra_outputs,_ = Graph_model(graphs)
                        predicts = predicts + outputs.cpu().detach().numpy().tolist()
                        gra_predicts = gra_predicts + gra_outputs.cpu().detach().numpy().tolist()
                    predicts = torch.tensor(predicts)
                    gra_predicts = torch.tensor(gra_predicts)
                    
                    # Print Acc. or MSE on validation set
                    if args.task_type == 'Classification':
                        valid_acc = accuracy(predicts, valid_label).item()
                        if valid_acc_saved < valid_acc:
                            valid_acc_saved = valid_acc
                            print('Epoch:',epoch+1)
                            print('Valid Performance:',valid_acc)
                        torch.save(Seq_model.state_dict(),args.model_path+'{}_Cont_Seq.pt'.format(args.dataset))

                    elif args.task_type == 'Regression':
                        valid_mse = loss_mse(predicts*(args.label_max-args.label_min),valid_label*(args.label_max-args.label_min)).item()
                        if valid_mse_saved > valid_mse:
                            valid_mse_saved = valid_mse
                            print('Epoch:',epoch+1)
                            print('Valid Performance:',valid_mse)
                            torch.save(Seq_model.state_dict(),args.model_path+'{}_Cont_Seq.pt'.format(args.dataset))
    
    #---------------- Test Phase----------------#

    Seq_model_load = Transformer(args).to(args.device)
    Seq_model_load.load_state_dict(torch.load(args.model_path+'{}_Cont_Seq.pt'.format(args.dataset)))
    Seq_model_load.eval()
    
    with torch.no_grad():
        predicts = []
        for graphs,sequences,labels in test_loader:
            outputs,_ = Seq_model_load(sequences)
            predicts = predicts + outputs.cpu().detach().numpy().tolist()
        predicts = torch.tensor(predicts)

    df_test = pd.read_csv('seq dataset/{}/test.csv'.format(args.dataset))

    if args.task_type == 'Classification':
        predict_tensor = predicts.max(1)[1].type_as(test_label)
        predict = predict_tensor.cpu().detach().numpy().tolist()

        df_test_save = pd.DataFrame()
        labels = test_label.squeeze(1).tolist()
        df_test_save['feature'] = df_test['Feature']
        df_test_save['predict'] = predict
        df_test_save['label'] = labels
        test_acc = accuracy(predicts,test_label).item()
        df_test_save['Acc'] = test_acc

        if args.output_layer > 2:
            from sklearn.metrics import precision_score, recall_score, f1_score
            df_test_save['Precision'] = precision_score(test_label.cpu(),predict_tensor.cpu().detach(),average='weighted')
            df_test_save['Recall'] = recall_score(test_label.cpu(),predict_tensor.cpu().detach(),average='weighted')
            df_test_save['F1-score'] = f1_score(test_label.cpu(),predict_tensor.cpu().detach(),average='weighted')

        if not os.path.exists('results/{}/co-modeling contrastive'.format(args.dataset)):
            os.mkdir('results/{}/co-modeling contrastive'.format(args.dataset))
        df_test_save.to_csv('results/{}/co-modeling contrastive/contrastive_seqlr{}_gralr{}_d{}_seed{}.csv'.format(args.dataset,args.seq_lr,args.graph_lr,args.d_model,args.seed))

    if args.task_type == 'Regression':
        predict = predicts.squeeze(1).cpu().detach().numpy().tolist()
        df_test_save = pd.DataFrame()
        labels = test_label.squeeze(1).tolist()
        df_test_save['feature'] = df_test['Feature']
        df_test_save['predict'] = predict
        df_test_save['label'] = labels
        error = []
        for i in range(len(labels)):
            error.append((labels[i]-predict[i])*(args.label_max-args.label_min))
        absError = []
        squaredError = []
        for val in error:
            absError.append(abs(val))
            squaredError.append(val*val)
        MSE = sum(squaredError)/len(squaredError)
        MAE = sum(absError)/len(absError)

        from sklearn.metrics import r2_score
        R2 = r2_score(test_label.cpu(),predicts.cpu().detach())
        
        df_test_save['MSE'] = squaredError
        df_test_save['MAE'] = absError
        df_test_save['MSE_ave'] = MSE
        df_test_save['MAE_ave'] = MAE
        df_test_save['R2'] = R2

        if not os.path.exists('results/{}/co-modeling contrastive'.format(args.dataset)):
            os.mkdir('results/{}/co-modeling contrastive'.format(args.dataset))
        df_test_save.to_csv('results/{}/co-modeling contrastive/contrastive_seqlr{}_gralr{}_d{}_nce{}.csv'.format(args.dataset,args.seq_lr,args.graph_lr,args.d_model,args.nce_weight))

if __name__ == '__main__':
    main()