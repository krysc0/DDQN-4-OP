import torch #nn
from torch_geometric import nn
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader

class GNN(torch.nn.Module):
    '''
    Encode the graph of the current orienteering tour

    '''
    def __init__(self, hyparams):
        super(GNN, self).__init__()
        self.layer1 = nn.GATConv(hyparams["input_dimension"], hyparams["h"][0], dropout=hyparams["dropout"])
        self.bnorm1 = torch.nn.BatchNorm1d(hyparams["h"][0])   
        self.layer2 = nn.GATConv(hyparams["h"][0], hyparams["h"][1], dropout=hyparams["dropout"])            
        self.bnorm2 = torch.nn.BatchNorm1d(hyparams["h"][1])

    def reset_parameters(self):
        self.layer1.reset_parameters()
        self.layer2.reset_parameters()
        self.bnorm1.reset_parameters()
        self.bnorm2.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        out = self.layer1(x,edge_index,edge_attr)
        out = self.bnorm1(out)
        out = torch.nn.functional.relu(out)
        out = self.layer2(out,edge_index,edge_attr)
        out = self.bnorm2(out)
        out = torch.nn.functional.relu(out)
        out = torch.sigmoid(out)
        return out
    
class DQN(torch.nn.Module):
    '''
    Estimate the Q-function given the current state and return q-values for actions

    Parameters
    ----------
    hyparams: dict
        v: int
            size of the instances
        input_dimension: int
            number of the graph's nodes' features 
        h: list[int]
            dimension of output of hidden layers (3)
        dropout: float
            probability that input features are randomly dropped
    x: torch.tensor
        node features [m x v x 3]
    edge_index: torch.tensor
        edge indices [m x 2 x num_edges]
    edge_attr: torch.tensor
        edge features [m x num_edges x 1]
    batch: torch.tensor
    
    '''
    def __init__(self, hyparams):
        super(DQN, self).__init__()
        self.node_embedding = GNN(hyparams)
        self.global_pool = nn.global_mean_pool
        self.linear1 = torch.nn.Linear(hyparams["h"][1], hyparams["h"][2])
        self.linear2 = torch.nn.Linear(hyparams["h"][2], hyparams["v"])

        torch.nn.init.uniform_(self.linear1.weight)
        torch.nn.init.uniform_(self.linear2.weight)
    
    def forward(self, x, edge_index, edge_attr, batch):
        out1 = self.node_embedding(x, edge_index, edge_attr)
        out2 = self.global_pool(out1, batch=batch)
        out3 = self.linear1(out2)
        out3 = torch.nn.functional.relu(out3)
        out4 = self.linear2(out3)
        q_estimate = torch.nn.functional.relu(out4)
        return q_estimate