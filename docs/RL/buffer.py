from collections import namedtuple, deque
import random
from utils import getHyparameters

class replayBuffer():
    '''
    Replay buffer of transitions stored in deque with maximum length b

    '''
    def __init__(self):
        hyparams = getHyparameters()
        self.buffer = deque(maxlen= hyparams["b"])

    def add(self,transition: namedtuple):
        ''' 
        Add a new transition to the buffer
        
        Parameter
        ---------
        transition: namedtuple
            Stores the (state, action, reward, next state, terminated, truncated, instance)          
        
        '''
        self.buffer.append(transition)

        return 0

    def sample(self, m):
        '''
        Sample a minibatch from the buffer

        Parameters
        ----------
        m: int
            mini batch size
        
        Return
        ----------
        mini_batch: list[namedtuple]

        '''
        sampled_indices = sorted(random.sample(range(len(self.buffer)), m) ,reverse=True)
        mini_batch = [self.buffer[i] for i in sampled_indices]
        #reverse sort indices so the element's index remains unchanged if elements after it are removed
        for j in sampled_indices: 
            del self.buffer[j]

        return mini_batch