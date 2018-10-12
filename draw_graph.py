# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 11:15:52 2018

@author: yishu
"""

import networkx as nx
import numpy as np

n_doc = 2
n_sent = 3
n_nodes = n_doc*n_sent

node_list = ['d'+str(i)+'s'+str(j) for i in np.arange(1, n_doc+1) for j in np.arange(1, n_sent+1)]
G = nx.complete_graph(node_list)

np.random.seed(seed=29)
random_weights = list(np.random.randn(n_nodes*(n_nodes-1)//2))
for i, edge in enumerate(G.edges()):
    G[edge[0]][edge[1]]['weight']=random_weights[i]
      
        
pos = nx.circular_layout(G)

edges = G.edges()

weights = [G[u][v]['weight']*2 for u,v in edges]

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 8, 8
import matplotlib.pyplot as plt
plt.clf()
#nx.draw_networkx(G, pos, with_labels=True, edges=edges, width=weights, node_color='c', node_shape="o", node_size=1000,  font_weight='heavy')        
nx.draw(G, pos, edges=edges, width=weights, node_color='c', node_shape="o", node_size=3000)        
nx.draw_networkx_labels(G, pos, font_weight='heavy', font_size=18)
#pos_higher = {}
#y_off = 0.1  # offset on the y axis
#
#for k, v in pos.items():
#    pos_higher[k] = (v[0], v[1]+y_off)
#    
#nx.draw_networkx_labels(G, pos_higher)    
