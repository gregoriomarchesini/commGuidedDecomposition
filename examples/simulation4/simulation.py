import numpy as np
import sys,os
from  networkx import Graph
from   stldec.decomposition_module import *
from   stldec.predicate_builder_module import *
from   stldec.control_module import *
from   stldec.visualization_module import simulateAgents


# Set folder directory.
results_dir = "/results"
results_dir =os.path.dirname(os.path.abspath(__file__)) + results_dir 
os.makedirs(results_dir, exist_ok=True)

orig_stdout = sys.stdout
f = open(results_dir+"/analysis.txt", 'w')
sys.stdout = f

# Define optimization problem 
#########################################################################################################
# Graph creation
#########################################################################################################
# create a set of node positions 

v1 = np.array([ 0.,0])
v2 = np.array([-20.,5.])
v3 = np.array([ 20.,5.])



# select maximum communication distance
maximumInterRobotDistance  = 8               # not binding for now
problemDimension           = len(v1)           # dimension of the problem in R^n

# create an iterable with index and attributes of the nodes you want to create

nodes =[(1,{"pos":v1}),
        (2,{"pos":v2}),
        (3,{"pos":v3})]


communicatingEdges = [(1,2),(1,3)]
nonCommunicatingEdges = [(2,3)]

edges = []
for i,j in communicatingEdges :
    edges.append((i,j,{"edgeObj":GraphEdge(source=i,target=j,isCommunicating = 1)}))
for i,j in nonCommunicatingEdges :
    edges.append((i,j,{"edgeObj":GraphEdge(source=i,target=j,isCommunicating = 0)}))


xx = [node[1]["pos"][0] for node in nodes]
yy = [node[1]["pos"][1] for node in nodes]
xxmin,xxmax = min(xx)*1.6,max(xx)*1.6
yymin,yymax = min(yy)*1.6,max(yy)*1.6

# create the whole graph for your system with task and communication edges
MASgraph = Graph()
MASgraph.add_nodes_from(nodes)
MASgraph.add_edges_from(edges)


#########################################################################################################
# Predicate construction
#########################################################################################################

# formation task
formula   = STLformula(temporalOperator   = "always",
                           predicate      = ellipsoidPredicate(center=np.array([0,0]),P = np.eye(2)),
                           timeinterval   = timeInterval(10.,15.))
formula.predicate.addSourceTarget(source=2,target=3) # if you don't specify it , it will take the source,target from the edge
MASgraph.edges[2,3]["edgeObj"].addFormula(formula)






initialTaskGraph,finalTaskGraph,commGraph  = computeNewTaskGraph(MASgraph=MASgraph,problemDimension=2,maxInterRobotDistance= maximumInterRobotDistance)
visualizeGraphs(commGraph, initialTaskGraph, finalTaskGraph)
sys.stdout = orig_stdout
f.close()


initialAgentsState ={ 1:np.array([0,0]),
                      2:np.array([-10,11]),
                      3:np.array([-25,-30]),}

stateTrajectories = simulateAgents(finalTaskGraph,tEnd=28,tStart=0,initialAgentsState=initialAgentsState,cleaningTimes=[15.4])


try:
    os.mkdir("./simulation3")
except OSError as error:
    print(error) 

for agentIndex,trajectory in stateTrajectories.items() :       
        with open(f"simulation3/agent{agentIndex}.npy", 'wb') as f:
                np.save(f,trajectory)

        
plt.show()

    
    
    