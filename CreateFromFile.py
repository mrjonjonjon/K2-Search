#create graph from .gph file of "parent, node" pairs using networkx. strips newline chars

import networkx as nx

import random
def read_from_file(filename):
    G = nx.DiGraph()
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            parent, node = line.split(',')
            parent = parent.strip()
            node=node.strip()
            
            G.add_edge(parent,node)
    return G

def add_random_edges(net,p):
    for n1 in list(net.nodes):
        for n2 in list(net.nodes):
            if n1==n2:
                continue
            if random.uniform(0,1)<p:
                net.add_edge(n1,n2)
            if len(list(nx.simple_cycles(net)))>0:
                net.remove_edge(n1,n2)

def remove_random_edges(net,p):
    for n1 in list(net.nodes):
        for n2 in list(net.nodes):
            if n1==n2:
                continue
            if random.uniform(0,1)<p and (n1,n2) in net.edges:
                net.remove_edge(n1,n2)
