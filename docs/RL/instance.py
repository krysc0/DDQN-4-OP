import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import re
from scipy.spatial import distance

class instance():
    """
    This class reads data about OP instances from CSV files.

    Parameters
    ----------
    filepath: str 
        The path to the CSV file.
        
    show: bool, optional
        Flag indicating whether to plot the instance using Networkx. Defaults to False.

    Attributes
    ----------
    filename: str
        Basename of the .csv file.
    v: int
        Number of vertices to be visited (including the start and end)
    V: np.ndarray[int], shape=(n+2, 2)
        Set of all vertices (including the start and end)
    r: np.ndarray[float], shape=(n+2, 1)
        Reward agent accrues by visiting each vertex
    T_max: float
        Maximum time to travel to all vertices
    

    Notes
    -----
    Line 1 of the .csv file contains "Tmax P" where
        Tmax - available time budget per path
        P - number of paths
    Lines 2+ of .csv file each line contains information about a vertex
    Information is written as x y s where
        x - x coordinate (float)
        y - y coordinate (float)
        s - profit of location (float)
    First point is the starting point
    Second point is the ending point

    """

    def __init__(self, filepath: str, show: bool = False):
        self.filename = os.path.basename(filepath)

        #Open and read the files line by line
        f = open(filepath,"r")
        file_content = f.readlines()

        #Get the time limit in the instance from the first line of file
        line1 = file_content[0]
        self.T_max = int(re.split(r'\s', line1)[0])
        #Determine the size of the instance
        self.v = sum([1 for line in file_content[1:] if not line.isspace()])
        #Initialize variables with zeros
        self.V = np.zeros(shape=(self.v,2))
        self.s = np.zeros(shape=(self.v,1))
        self.t = np.zeros(shape=(self.v,self.v))

        for i,line in enumerate(file_content[1:]):
            if not line.isspace():
                split_line = re.split(r'\s+', line)
                split_line = [x for x in split_line if x != '']
                
                [x, y, s] = split_line                  
                self.V[i] = [float(x),float(y)]
                self.s[i] = float(s)
        
        f.close()

        #Calculate t[i][j] which is the time to travel between vertex i and j, assuming a speed of 1
        for i in np.arange(len(self.V)):
            for j in np.arange(len(self.V)):
                if i != j:
                    dist_ij = distance.euclidean(self.V[i], self.V[j])
                    self.t[i][j] = dist_ij

        #Create dictionary to map the index of each node to its coordinates. Will use in render
            self.C={}
            for i,c in enumerate(self.V):
                self.C[i] = c

        #save the index of the vertices where the tour will start an end
        self.start_vertex = 0 
        self.end_vertex = 1
        
        if show:

            #Show the instance with Networkx
            plt.figure()
            fig, ax = plt.subplots(1,figsize=(10,10))
            ax.set_title('OP Instance ' + self.filename)
            G = nx.Graph()
            G.add_nodes_from(range(self.v))
            nx.draw_networkx(G, pos=self.C, arrows=None,node_color='#FFC0CB',node_size=100,font_size=9)
            plt.show()