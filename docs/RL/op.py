from collections import namedtuple
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
from utils import getHyparameters, getTour

class op_env(gym.Env):
    '''
    Define the RL environment using gym

    Parameters
    ----------
    render_mode: str
        specify how we will view the environment during the creation of the buffer. Must be one of {'human','rgb_array'}
        * human : plot graphs with networkx
        * rgb_array : print the tours so far as arrays
    instances: list[obj]
        benchmark training instances objects
    
    Attributes
    ----------
    action_space: np.ndarray
        gymnasium.spaces.discrete object which will store the vertices that the agent visits in each graph
    observation_space: np.ndarray
        gymnasium.spaces.box object which will store the tour's adjacency matrix and current time 
    transition: namedtuple
        store state, action, reward, next state and termination status for each action taken by the agent

    Notes
    -----
        * there are v potential vertices an agent can visit so there are v discrete actions
        * episode ends once all the nodes have been visited or the time has run out
        * state is deterministic so observation = space

    '''
    def __init__(self, instance):
        self.hyparams = getHyparameters()

        self.observation_space = {
            "time": spaces.Box(low = 0.0, high = np.inf , shape=(1,), dtype=np.float32),
            "A" : spaces.Box(low=0, high=1, shape=(self.hyparams["v"], self.hyparams["v"]),dtype=np.bool_)
        }
       
        self.action_space = spaces.Discrete(self.hyparams["v"])
        self.transition = namedtuple('Transition', ('state','action','reward','next_state','terminated','truncated','instance'))
        self.terminated = False
        self.truncated = False
        self.render_mode = "human"
        self.instance = instance

        self.reset()
      
    def reset(self):
        '''
        Reset adjacency matrix and rewards for all instances

        Returns
        -------
            initial_observation

        '''
        self.observation = {}
        self.observation["time"] = 0.0 
        self.observation["A"] = np.zeros(shape=(self.instance.v, self.instance.v))

        return self.observation
          
    def step(self, action):
        '''
        Move agent from current vertex to another vertex. Update the environment.

        Parameters
        ----------
        action: int
            action to be taken to update the environment state

        Returns
        -------
        transition: namedTuple
            (state, action, reward, next state, terminated, truncated, instance)
        '''

        observation0 = self.observation
        s_next = action
        s_current = getTour(self.instance, observation0)[-1]
        t_current = self.observation["time"]
        t_next = t_current + self.instance.t[s_current][s_next]
        #print(f" Current state {s_current} next state {s_next} current time {t_current} next time {t_next}")

        if t_next <= self.instance.T_max:
            self.observation["A"][s_current][s_next] = 1
            self.observation["time"] = t_next
            if s_next == self.instance.end_vertex:
                self.terminated = True #got to the end node
                self.truncated = False
                reward = 0 #no reward
            else:
                self.terminated = False
                self.truncated = False
                reward = self.instance.s[s_next][0]
        else:
            self.truncated = True #ran out of time
            self.terminated = False
            reward = 0         
        
        transition = self.transition(state=observation0,action=action,reward=reward,next_state=self.observation,\
                                     terminated=self.terminated,truncated=self.truncated,instance=self.instance)
        return transition
    
    def render(self):
        '''
        Plot the tour so far or print the environment state

        '''
        # if self.render_mode=="graph":
        #     fig, ax = plt.subplots(1,figsize=(10,10))
        #     ax.set_title('OP Instance ' + self.filename)
        #     G = nx.Graph()
        #     G.add_nodes_from(range(self.instance.v))
        #     for i in range(self.instance.v):
        #         for j in range(self.instance.v):
        #                 if self.observation["A"][i][j] == 1:
        #                     G.add_edge(i, j)
        #     nx.draw_networkx(G, pos=self.instance.C, arrows=None,node_color='#FFC0CB',node_size=100,font_size=9)
        #     plt.show()

        os.system("cls")
        visited = getTour(self.instance, self.observation)
        print(f"Tour so far {visited}")

        return 0