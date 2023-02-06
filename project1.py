from collections import defaultdict
from multiprocessing import Pool
import sys
import json
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import random
import sys
from graphviz import Source
import time
sys.path.append('/Users/jonathanakaba/Desktop/AAHW1/project1')
from networkx.drawing.nx_agraph import write_dot
from localsearch import *
from localsearch.OpTypes import OpType
from localsearch.GraphOp import *
from localsearch.GraphOperations import Add,Remove,Reverse
from localsearch.GraphOpFactory import *
from utils.CreateFromFile import read_from_file,add_random_edges,remove_random_edges
import itertools


def swaprandom(l,numswaps=1):
    for q in range(numswaps):
        i=random.randrange(0,len(l))
        j=random.randrange(0,len(l))
        j=(i+1)%len(l)
        if i==j:
            continue
        t=l[i]
        l[i]=l[j]
        l[j]=t
        
def write_gph(dag, filename):
    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write("{}, {}\n".format(edge[0], edge[1]))

def build_M(data,num_instantiations,net):
    M={}
    for colname,col in data.items():
        build_M_i_vectorized(data,colname,net,M,num_instantiations)
    return M
ns=[]
def draw_graph(G:nx.DiGraph):

 
    pos = nx.kamada_kawai_layout(G)
    nx.draw_networkx(G,node_size=200,font_size=5,font_weight='bold',node_color='#FFFF00')
    plt.show()

def build_M_i_vectorized(data, varname, net, M, num_instantiations):
    predecessors = list(net.pred[varname].keys())
    parent_instantiations = data.loc[:, predecessors].to_numpy()
    child_instantiations = data[varname].to_numpy()
    
    unique_parents, parent_indices = np.unique(parent_instantiations, axis=0, return_inverse=True)
    M_i_dict = np.zeros((len(unique_parents), num_instantiations[varname]), dtype=int)
    
    np.add.at(M_i_dict, (parent_indices, child_instantiations - 1), 1)
    
    M[varname] = pd.DataFrame(M_i_dict, index=map(tuple, unique_parents))


def bayesian_score(M,varnames,num_instantiations,net):
    
    score=0
    for var in varnames:
        score+=bayesian_score_component(var,M,num_instantiations,net)
    return score


def bayesian_score_component(varname,M,num_instantiations,net):
    M_i=M[varname]
    p=M_i.copy().applymap(lambda x:x+1).applymap(math.lgamma).values.sum()

    predecessors = list(net.pred[varname].keys())

    num_inst_of_parents=pd.DataFrame([num_instantiations[p] for p in predecessors])
    q_i=num_inst_of_parents.values.prod()

    r_i=num_instantiations[varname]

    p -= math.lgamma(1)*r_i*q_i#constant
    
    p+=math.lgamma(r_i)*q_i

    p-=math.lgamma(r_i)*(q_i - len(M_i.index)) + M_i.copy().sum(axis=1).map(lambda x:x+r_i).map(math.lgamma).values.sum()
    return p 


def K2(data,num_instantiations,net,shuffle=False):
   
    max_score_increase=1
    num_iters=0
    ordering =list(net.nodes) 
    if shuffle:
        random.shuffle(ordering)
        
   
    print(f'STARTING SCORE: {bayesian_score(build_M(data,num_instantiations,net),list(data.columns),num_instantiations,net)}')
    for i1,n1 in enumerate(ordering):
        
        max_score_increase=1
        best_score=bayesian_score(build_M(data,num_instantiations,net),list(data.columns),num_instantiations,net)
        while(max_score_increase>0):
            num_iters+=1
            best_edge=0
            base_score=bayesian_score(build_M(data,num_instantiations,net),list(data.columns),num_instantiations,net)
            max_score_increase=-1
            for i2 in range(i1):
                n2=ordering[i2]
                if n1==n2:
                    continue
                if (n2,n1) in net.edges:
                    continue

                net.add_edge(n2,n1)

                if len(list(nx.simple_cycles(net)))>0:
                    net.remove_edge(n2,n1)
                    continue
                M=build_M(data,num_instantiations,net)
                newscore = bayesian_score(M,list(data.columns),num_instantiations,net)
                increase=newscore-base_score
                if increase>max_score_increase:
                    best_edge=(n2,n1)
                    max_score_increase=increase
                    
                net.remove_edge(n2,n1)

            if max_score_increase>0:
                net.add_edge(best_edge[0],best_edge[1])
                best_score=base_score+max_score_increase
                #print(f"NEW SCORE:{base_score+max_score_increase}")
            else:
                #print(f'score unchanged:{base_score}')
                pass
        
    return ordering,net,best_score

def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")
    M={}

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    bayesian_network=nx.DiGraph()
    
    data = pd.read_csv('data/'+inputfilename)
    nodes=list(data.columns)
    bayesian_network.add_nodes_from(data.columns)

    num_instantiations={}
    for colname,col in data.items():
        num_instantiations[colname]=max(col)
    
    t0=time.time()
    K2(data,num_instantiations,bayesian_network,False)
    t1=time.time()

    print(f"TOOK {t1-t0} SECONDS")
    write_gph(bayesian_network,outputfilename)
    print('score is ',bayesian_score(build_M(data,num_instantiations,bayesian_network),nodes,num_instantiations,bayesian_network))

    
    draw_graph(bayesian_network)




    
if __name__ == '__main__':

    main()
