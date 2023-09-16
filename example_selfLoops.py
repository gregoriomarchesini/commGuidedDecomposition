import networkx as nx

G : nx.Graph = nx.cycle_graph(7)
G.add_edge(1,1)
paths = list(nx.shortest_simple_paths(G, 0, 3))
print(paths)