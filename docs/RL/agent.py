import numpy as np
import torch
from op import op_env
from utils import getHyparameters, getTour, getBatchofOne, getBatchofMany
from buffer import replayBuffer
from model import DQN
from tqdm import tqdm

class Agent():
    '''
    Agent interacts with the environment and connects the environment and the model
    Initialize the environment, model and the hyparameters

    Parameters:
    device: str
        Either "cpu" or "cuda"

    '''
    def __init__(self, device):
        self.hyparams = getHyparameters()
        self.buffer = replayBuffer()
        self.offlineModel = DQN(self.hyparams)
        self.onlineModel = DQN(self.hyparams)
        self.epsilon = self.hyparams["initial_epsilon"]
        self.lossFN = torch.nn.MSELoss()
        self.offlineModel.to(device)
        self.onlineModel.to(device)

        self.optimizer = torch.optim.Adam(self.onlineModel.parameters(),\
                                           lr=self.hyparams["learning_rate"],\
                                            weight_decay= self.hyparams["l2"])
        
    def fillBuffer(self, instances):
        '''
        Fill the buffer with transitions from the instances stored in files
        According to some initial policy which uniformly selects actions

        '''
        print("Filling buffer...", end=" ")
        done = False
        with tqdm(total=self.hyparams["b"]) as pbar:
            while not done:
                self.resetEnvs(instances)
                for env in self.envs:
                    while not (env.terminated or env.truncated):
                        initial_policy = np.ones(env.action_space.n) / env.action_space.n

                        state = env.observation

                        action = self.choose_action(initial_policy, env.instance, state)
                        transition = env.step(action)

                        self.buffer.add(transition)
                        pbar.update(1)
                    #print(f"Tour: {getTour(env.instance, state)}")
                        if len(self.buffer.buffer) >= self.hyparams["b"]:
                            done = True
                            break

        return 0
       
    def choose_action(self, model_output, instance, state):
        '''
        Choose the action to be performed according to the output of the online model
        Action chosen with some probability according to epsilon-greedy rule
        '''
        visited = getTour(instance,state)
        actions = np.arange(instance.v)

        if np.random.random() > self.epsilon:
            masked_output = np.where(np.isin(actions, visited), -np.inf, model_output)
            action = np.argmax(masked_output) #Exploitation
        else:
            options = np.delete(actions, visited)
            action = np.random.default_rng().choice(options) #Exploration
        
        return action
    
    def decayEpsilon(self):
        ''' 
        Decay epsilon at the end of each episode

        '''
        if self.epsilon > self.hyparams["minimum_epsilon"]:
            self.epsilon = self.epsilon * self.hyparams["epsilon_decay"]
        
        return 0
    
    def resetEnvs(self, instances):
        ''' 
        Reset environments and objective of each instance

        '''
        self.envs = [op_env(instance=instance)\
                    for instance in instances]

        self.obj_fn = np.zeros(shape=(len(self.envs)))

        self.isDone = False

        return 0

    def checkEpisodeDone(self):
        '''
        If all of the environments are terminated or truncated then the episode is done and we decay epsilon
        
        '''
        self.isDone = np.all([np.any([env.terminated, env.truncated]) for env in self.envs])
        if self.isDone:
            self.decayEpsilon()    

    def collect(self, device: str):
        '''
        Use the online Q-network to select the actions
        Perform and save a transition from each instance's environment
        Once the environment is terminated or truncated we calculate the total rewards

        Parameters:
        devie: str
            Has only two possible values "cpu" or "cuda"
        '''
        self.onlineModel.eval()
        for i,env in enumerate(self.envs):
            state = env.observation
            batch = getBatchofOne(env.instance, state)
            batch.to(device)
            x = batch.x.float() 
            edge_index = batch.edge_index.type(torch.int64)
            edge_attr = batch.edge_attr.long()
            q_estimate = self.onlineModel(x, edge_index, edge_attr, batch.batch).type(torch.float64).to(device)
            policy = torch.nn.functional.softmax(q_estimate, dim=1)
            policy = policy.detach().numpy()
            
            action = self.choose_action(policy, env.instance, state)
            transition = env.step(action)
            visited = getTour(env.instance, transition.state)

            self.buffer.add(transition)

            if transition.terminated or transition.truncated:
                for vertex in visited:
                    self.obj_fn[i] += env.instance.s[vertex]
        
        self.checkEpisodeDone()
                   
        return 0

    def learn(self, device: str):
        '''
        Use the offline (target Q network to evaluate the actions (from the buffer)

        Perform a single gradient step for the online :
            * sample a mini batch of transitions
            * evaluate the current state to get the online Q value estimate for each transition's state
            * evaluate the next state to get the offline Q value estimate for each transition's state
            * calculat and backpropagate loss
        Parameters
        ----------
        device: str
            Either "cpu" or "cuda"

        Returns
        -------
        loss: float
            MSE loss
        
        '''
        self.onlineModel.train(True)
        mini_batch = self.buffer.sample(self.hyparams["m"])

        #GET BATCH OF DATA FOR CURRENT STATE AND NEXT STATE
        batch_s, batch_ns = getBatchofMany(mini_batch)
        batch_s.to(device)

        #USE CURRENT Q NETWORK TO PREDICT Q VALUE OF ACTIONS TAKEN IN CURRENT STATE
        rewards = torch.tensor([t.reward for t in mini_batch], dtype=torch.float32)
        terminated = torch.tensor([int(t.terminated) for t in mini_batch], dtype=torch.int32)
        actions = torch.unsqueeze(torch.tensor([t.action for t in mini_batch], dtype=torch.int64), 1)
 
        x = batch_s.x.float()
        edge_index = batch_s.edge_index.type(torch.int64)
        edge_attr = batch_s.edge_attr.long()
        q_estimate = self.onlineModel(x, edge_index, edge_attr,batch_s.batch).type(torch.float64).to(device) #shape:32x66
        predicted_q = torch.gather(q_estimate, dim=1, index=actions)                   
        #USE TARGET Q NETWORK TO EVALUATE VALUE OF ACTIONS TAKEN IN NEXT STATE
        x = batch_ns.x.float()
        edge_index = batch_ns.edge_index.type(torch.int64)
        edge_attr = batch_ns.edge_attr.long()
        next_state_output = self.offlineModel(x, edge_index, edge_attr,batch_ns.batch).type(torch.float64)

        #Expected value of next state under phi' ~ max Q of actions that can be taken next state under phi' ()
        next_state_expected_value, _ = torch.max(next_state_output, dim=1)
        #Estimate Q value of current state using next state under phi
        #Q value of termination state is always 0
        target_q = rewards + (self.hyparams["gamma"] * (1 - terminated) * next_state_expected_value)
        target_q = torch.unsqueeze(target_q, dim=1)
        #calculate loss + backward prop
        loss = self.lossFN(predicted_q, target_q)
        loss.backward()
        self.optimizer.step()

        return loss

    def copyModelParams(self):
        '''
        At the end of an episode copy the online model parameters to the offline model
        
        '''
        self.offlineModel.train(False)
        self.onlineModel.train(False)
        self.offlineModel.load_state_dict(self.onlineModel.state_dict())