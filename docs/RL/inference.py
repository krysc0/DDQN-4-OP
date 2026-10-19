from agent import Agent
from utils import getTour, getBatchofOne
from op import op_env
import numpy as np
import torch

class TestAgent(Agent):
    def __init__(self, device):
        super().__init__(device)
        
    def __call__(self, agent, instances, device):
        '''
        Replace the collect method from training agent
        
        '''
        self.resetEnvs(instances)
        agent.onlineModel.eval()
        self.epsilon = 0
        for i,env in enumerate(self.envs):
            while not (env.terminated or env.truncated):
                state = env.observation
                batch = getBatchofOne(env.instance, state)
                batch.to(device)
                x = batch.x.float() 
                edge_index = batch.edge_index.type(torch.int64)
                edge_attr = batch.edge_attr.long()
                q_estimate = agent.onlineModel(x, edge_index, edge_attr, batch.batch).type(torch.float64).to(device)
                policy = torch.nn.functional.softmax(q_estimate, dim=1)
                policy = policy.detach().numpy()
                action = self.choose_action(policy, env.instance, state)
                transition = env.step(action)
                
            visited = getTour(env.instance, transition.state)
            for vertex in visited:
                self.obj_fn[i] += env.instance.s[vertex]
                         
        return 0
