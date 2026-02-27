len(T)
23369
type(T.nodes())
#####################
# Import necessary modules
import matplotlib.pyplot as plt
import networkx as nx

# Draw the graph to screen
nx.draw(T_sub)
plt.show()
########################
# Use a list comprehension to get the nodes of interest: noi
noi = [n for n, d in T.nodes(data = True) if d['occupation'] == 'scientist']

# Use a list comprehension to get the edges of interest: eoi
eoi = [(u, v) for u, v, d in T.edges(data = True) if d['date'] < date(2010 , 1 , 1)]
################
type(T)
########################
# Set the weight of the edge
T.edges[1 , 10]['weight'] = 2

# Iterate over all the edges (with metadata)
for u , v , d in T.edges(data = True):

    # Check if node 293 is involved
    if 293 in [u , v]:

        # Set the weight to 1.1
        T.edges[u , v]['weight'] = 1.1
###############################
# Define find_selfloop_nodes()
def find_selfloop_nodes(G):
    """
    Finds all nodes that have self-loops in the graph G.
    """
    nodes_in_selfloops = []

    # Iterate over all the edges of G
    for u, v in G.edges():

    # Check if node u and node v are the same
        if u == v:

            # Append node u to nodes_in_selfloops
            nodes_in_selfloops.append(u)

    return nodes_in_selfloops

# Check whether number of self loops equals the number of nodes in self loops
assert nx.number_of_selfloops(T) == len(find_selfloop_nodes(T))
