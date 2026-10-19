import argparse
import numpy as np
import time
import torch
from tqdm import tqdm
from agent import Agent
from inference import TestAgent
from utils import getHyparameters, getInstances, checkHyparams, earlyStop, saveResults
from torch.utils.tensorboard import SummaryWriter

def run_training(hyparams: dict = getHyparameters()):
    '''
    Train agents to complete a path within the time limit to maximize their combined reward

    Parameters
    ----------
    hyparams: dict, optional
        v: int
            size of the instances
        input_dimension: int
            number of the graph's nodes' features 
        h: list[int]
            dimension of output of hidden layers (3)
        dropout: float
            probability that input features are randomly dropped
        learning_rate: float
            Learning rate to use for training
        l2: float
            L2 regularization rate
        initial_epsilon: float
            Initial value of epsilon to be used for exploration of the environment
        m: int
            Size of mini batch (number of transition observations i) to sample from buffer
        max_episodes: int
            The number of episodes used in training
        epsilon_decay : float
            Rate to decay epsilon at after each episode
        minimum_epsilon: float
            Minimum value of epsilon
        gamma: float
            Factor by which we reduce epsilon each episode
        k: int
            Number of timesteps taken with current batch before we update policy network (default = 1)
        n: int
            Maximum number of timesteps to take before we update target network
        b: int
            Maximum number of transitions to store in the buffer memory
        delta: float
            Minimum deviation in loss in a given number of consecutive episodes
        patience: int
            Number of episods to wait before concluding loss has converged
        
    Returns
    ----------
    results: dict
        Maps the filename of each instance to the reward gotten during training

    Notes
    -----
    see Algorithm.png for details about the learning process

    '''
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Device: {device}")
    
    training_instances, validation_instances, testing_instances = getInstances()
    print(f"Number of instances used in training: {len(training_instances)}")
    hyparams = checkHyparams(training_instances)

    agent = Agent(device)
    val_agent = TestAgent(device)
    earlystop = earlyStop()
    writer = SummaryWriter()
    converged = False
    i=0
    agent.fillBuffer(training_instances)
    while not converged:
        for _ in tqdm(range(hyparams["n"])):
            ep_start_t = time.time()
            #---------- EPISODE------------------
            total_loss = 0
            agent.resetEnvs(training_instances)
            t = 0
            while not agent.isDone: 
                agent.collect(device)
                t+=1
            for _ in range(hyparams["k"]): 
                total_loss += agent.learn(device)
            #---------------- EPISODE--------------
            avg_loss = total_loss / (t+1)

            writer.add_scalar("loss/training", avg_loss, i+1)
            writer.add_scalar("episode_duration", t+1 , i+1)
            writer.add_scalar("episode_time", time.time() - ep_start_t, i+1)

            for e, instance in enumerate(training_instances):
                writer.add_scalar(f"objective/train/{instance.filename}", agent.obj_fn[e], i+1)

            val_agent(agent, validation_instances, device)

            for e, instance in enumerate(validation_instances):
                writer.add_scalar(f"objective/val/{instance.filename}", val_agent.obj_fn[e], i+1)
            
            if earlystop(avg_loss):
                print(f"Training converged at episode {i+1}")
                converged = True
                break
            
            i+=1
        #-------------EVERY N EPISODES-------------------------
        agent.copyModelParams
        print(i,len(agent.buffer.buffer))

        if i >= hyparams["max_episodes"]:
            print(f"Max number of episodes reached. Ending training after {i} episodes")
            break
    
    #----------------AFTER TRAINING IS DONE----------------- 
    test_agent = TestAgent(device)
    test_agent(agent, testing_instances, device)
    saveResults(testing_instances, test_agent.obj_fn)

    return 0

if __name__ == '__main__':
    run_training()