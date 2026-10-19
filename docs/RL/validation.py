from agent import Agent
from utils import getBatchofMany
import numpy as np
import torch

class ValAgent(Agent):
    def __init__(self):
        super().__init__()
        
    def __call__(self, agent, instances):
        self.fillBuffer(instances)
        
        agent.onlineModel.train(False)
        mini_batch = self.buffer.sample(self.hyparams["m"])

        #GET BATCH OF DATA FOR CURRENT STATE AND NEXT STATE
        batch_s, batch_ns = getBatchofMany(mini_batch)

        #USE CURRENT Q NETWORK TO PREDICT Q VALUE OF ACTIONS TAKEN IN CURRENT STATE
        rewards = torch.tensor([t.reward for t in mini_batch], dtype=torch.float32)
        terminated = torch.tensor([int(t.terminated) for t in mini_batch], dtype=torch.int32)
        actions = torch.unsqueeze(torch.tensor([t.action for t in mini_batch], dtype=torch.int64), 1)
 
        x = batch_s.x.float()
        edge_index = batch_s.edge_index.type(torch.int64)
        edge_attr = batch_s.edge_attr.long()
        q_output = self.onlineModel(x, edge_index, edge_attr,batch_s.batch).type(torch.float64) #shape:32x66
        predicted_q = torch.gather(q_output, dim=1, index=actions)                   
        #USE TARGET Q NETWORK TO EVALUATE VALUE OF ACTIONS TAKEN IN NEXT STATE
        x = batch_ns.x.float()
        edge_index = batch_ns.edge_index.type(torch.int64)
        edge_attr = batch_ns.edge_attr.long()
        next_state_output = agent.offlineModel(x, edge_index, edge_attr,batch_ns.batch).type(torch.float64)

        #Expected value of next state under phi' ~ max Q of actions that can be taken next state under phi' ()
        next_state_expected_value, _ = torch.max(next_state_output, dim=1)
        #Estimate Q value of current state using next state under phi
        #Q value of termination state is always 0
        target_q = rewards + (self.hyparams["gamma"] * (1 - terminated) * next_state_expected_value)
        target_q = torch.unsqueeze(target_q, dim=1)
        #calculate loss + backward prop
        loss = self.lossFN(predicted_q, target_q)

        return loss