import json
import numpy as np
import glob
import torch
from sklearn.model_selection import train_test_split
from instance import instance
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

def getInstances(train_split : float = 0.7,  val_split: float = 0.15,  test_split: float = 0.15):
    '''
    Create a list of Instance class objects that will be used for training, testing and validation
    
    Parameters
    ----------
    train_split, val_split, test_split: float, float, float
        Percentage of instances to use in neural network model 
    split: str, optional
        Which of the instances should be returned. Choices=["train","val","test"]
        Default = "train"
    
    Returns
    -------
        train_instances, val_instances, test_instances: list[Instance]
    
    Raises
    ------
        ValueError: If the sum of ratios is not 1.0.

    '''

    directory = f"Instances/set_66_1"
    files = glob.glob(directory+"/*")

    if not files:
        raise ValueError(f"Directory '{directory}' is empty or does not exist.")

    if abs(np.sum([train_split, val_split, test_split]) - 1) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0. Please adjust the ratios.")

    train_files, p = train_test_split(files, test_size= val_split + test_split, random_state=42)
    test_files, val_files = train_test_split(p, test_size= val_split / (val_split + test_split), random_state=42)

    train_instances = [instance(filepath = file, show=False) for file in train_files]
    val_instances = [instance(filepath = file, show=False) for file in val_files]
    test_instances = [instance(filepath = file, show=False) for file in test_files]

    return train_instances, val_instances, test_instances


def getHyparameters():
    '''
    Load the hyparameters from a json file

    Return
    ------
    hyparameters: dict
        dictionary storing the hyparameters

    '''
    file_path = f"hyparams.json"
    f = open(file_path,) 
    hyparameters = json.load(f)
    f.close()
    return hyparameters

def setHyparameters(hyparameters):
    '''
    Dump the hyparameters to a json file

    Parameter
    ------
    hyparameters: dict
        dictionary storing the hyparameters

    '''
    file_path = f"hyparams.json"
    f = open(file_path,"w") 
    json.dump(hyparameters, f)
    f.close ()
    return 0

def getStaticFeatures(instance):
    '''
    Get the score of each vertex
    '''
    x0 = instance.s.flatten()
    return x0


def getTour(instance, state):
    '''
    Get the path the agent has visited so far in this episode
    '''
    done=False
    visited = np.array([instance.start_vertex])
    if np.sum(state["A"]) > 0:
        while not done:
            for i, row in enumerate(state["A"]):
                if visited[-1] == i:
                    if np.all(row == 0):
                        done = True
                    else:
                        visited = np.append(visited, np.where(row==1)[0][0])
                    break

    return visited

def getDynamicFeatures(instance, state):
    '''
    Parameter
    ---------
    transition: namedtuple
        (state, action, reward, next state, terminated, truncated, instance)
        where state has keys:
        * "time": float
        * "A" : (v,v) np.ndarray

    Return
    ------
        x1: float
            Fraction of time elapsed since tour has begun if we travel to the node at current time
        x2: float
            Fraction of time left till tour must end if we travel to the node at current time

    '''
    T_max = instance.T_max
    v = instance.v
    t = instance.t

    t_elapsed = state["time"] / T_max
    t_remaining = 1.0 - t_elapsed
    current_vertex = getTour(instance, state)[-1]

    x1 = np.zeros(shape= v)
    x2 = np.zeros(shape= v)
    for i, time in enumerate(t[current_vertex]):
        x1[i] = t_elapsed + (time / T_max)
        x2[i] = 1 - x1[i]

    return x1, x2

def getGraphDataObj(instance, state):
    '''
    Create a torch_geometric.data object from the transition

    Return
    ------
    x: 
        node feature matrix (num_nodes, num_features)
    ei:
        graph conectivity with edges listed in ((to)(from)) format. shape=(2, num_edges)
    ea:
        edge feature matrix (num_edges, num_edge_features)
    
    '''
    T_max = instance.T_max
    t = instance.t

    x = np.zeros(shape=(instance.v,3), dtype = np.float64)
    x[:,0] = getStaticFeatures(instance)
    x[:,1], x[:,2] = getDynamicFeatures(instance, state)

    visited = getTour(instance, state) #has maximum length = no of vertices + 1

    edge_indices = np.zeros(shape=(len(visited)-1,2),dtype=np.int64)
    edge_attributes = np.zeros(shape=(len(visited)-1,1),dtype=np.float64)
    
    for i,node in enumerate(visited):
        if i < len(visited) - 1:
            edge_indices[i] = tuple([node,visited[i+1]])
    
    for j, pair in enumerate(edge_indices):
        [fr, to] = pair
        edge_attributes[j] = t[fr][to] / T_max

    edge_indices = np.ascontiguousarray(np.transpose(edge_indices))
    x, edge_indices, edge_attributes = torch.tensor(x), torch.tensor(edge_indices), torch.tensor(edge_attributes)

    data = Data(x = x, edge_index= edge_indices, edge_attr=edge_attributes)
    
    return data

def getBatchofOne(instance, state):
    '''
    Use dataloder to get a batch (of size 1) for the pyg data
    '''
    data = [getGraphDataObj(instance, state)]
    dataloader = DataLoader(data, batch_size=1)
    batch = next(iter(dataloader))
    return batch

def getBatchofMany(mini_batch):
    '''
    Use dataloder to get a mini-batch (of size m) for the pyg data
    '''
    hyparams = getHyparameters()

    data_list1 = [getGraphDataObj(t.instance, t.state) for t in mini_batch]
    dataloader1 = DataLoader(data_list1, batch_size= hyparams["m"], shuffle = True) 
    batch_state = next(iter(dataloader1))

    data_list2 = [getGraphDataObj(t.instance, t.next_state) for t in mini_batch]
    dataloader2 = DataLoader(data_list2, batch_size= hyparams["m"], shuffle = True) 
    batch_next_state = next(iter(dataloader2))

    return batch_state, batch_next_state

def checkHyparams(instances):
    '''
    Check if all the instances being used for train, validation or test are of the same size
    Check if the hyperameters stored matches this size

    Parameters
    ----------
    instances: list[object]
        list of instance objects
    
    Raise
    -----
    SystemExit 
        if the instances are not of the same size
    ValueError
        if the hyparameter v does not match the size of the instances
    
    '''

    sizes = [instance.v for instance in instances]
    if not np.all([size == sizes[0] for size in sizes]):
        raise SystemExit("Training instances are not all of the same size. Stopping the code.")

    hyparams = getHyparameters()

    try:
        valid = (hyparams["v"] == sizes[0])
        if not valid:
            raise ValueError("The size of the training instances is not the same as v")
    
    except ValueError as e:
        print(e)
        print(f"Setting v which is currently {hyparams['v']} to be the size of the training instances {sizes[0]}")
        hyparams["v"] = sizes[0]
        setHyparameters(hyparams)
    
    return hyparams

class earlyStop():
    '''
    Object to call on to determine whether to early stop training
    '''
    def __init__(self):
        hyparams = getHyparameters()
        self.patience = hyparams["patience"]
        self.delta = hyparams["delta"]
        self.wait = 0
        self.best_loss = None
        self.last_loss = None

    def __call__(self, current_loss):
        if self.best_loss is None:
            self.best_loss = current_loss
            self.last_loss = current_loss
        elif self.best_loss > current_loss: #current loss is better than the best loss
            self.best_loss = current_loss
            self.last_loss = current_loss
            self.wait = 0
        elif self.last_loss - current_loss < self.delta: #loss is increasing consinstently/not changing signficantly
            self.last_loss = current_loss    
            self.wait += 1
            if self.wait >= self.patience:
                return True  # Stop training
        else: #loss is decreasing as it should
            self.wait = 0
        return False  # Continue training
    
def getResults(instances, obj_fn):
    '''
    Dump the results to a json file

    Parameters
    instances: list[obj]
        List of OP instance objects
    obj_fn: list
        Stores the rewards reached for each instance

    Return
    ------
    results: dict
        dictionary storing the objective function for each instance

    '''
    results = {}
    for e, instance in enumerate(instances):
        results[instance.filename] = obj_fn[e]

    return results

def saveResults(instances, obj_fn):
    '''
    Save the test results to a .json file
    '''
    results = getResults(instances, obj_fn)

    file_path = f"test_results.json"
    f = open(file_path,"w") 
    json.dump(results, f)
    f.close ()

    return results