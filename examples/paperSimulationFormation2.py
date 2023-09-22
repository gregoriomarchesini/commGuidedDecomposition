import numpy as np
import casadi as ca
import networkx as nx
from   multiagent_STLdec.decomposition_module import *
from  multiagent_STLdec.predicate_builder_module import *
from multiagent_STLdec.control_module import *
from multiagent_STLdec.visualization_module import simulateAgents
import sys,os

orig_stdout = sys.stdout
f = open('simulation2_decompositon.txt', 'w')
sys.stdout = f


# define optimization problem 

#########################################################################################################
# Graph creation
#########################################################################################################
# create a set of node positions 

v1 = np.array([0,0])
v2 = np.array([10,0])
v3 = np.array([10,10])
v4 = np.array([10,0])

# select maximum communication distance
maximumInterRobotDistance  = 100000                # not binding for now
problemDimension           = len(v1)             # dimension of the problem in R^n
distPred :convexPredicateFunction     = maxDistancePredicate(stateSpaceDimension=problemDimension,maxDistance=maximumInterRobotDistance) # here the target and source do not matter really
maxDistanceFunction = distPred.function

# create an iterable with index and attributes of the nodes you want to create

nodes =[(1,{"pos":v1}),
        (2,{"pos":v2}),
        (3,{"pos":v3}),
        (4,{"pos":v4})]

edges = [(1,2,{"edgeObj":GraphEdge(source=1,target=2,isCommunicating = 1)}),
         (2,3,{"edgeObj":GraphEdge(source=2,target=3,isCommunicating = 1)}),
         (3,1,{"edgeObj":GraphEdge(source=3,target=1,isCommunicating = 1)}),
         (4,2,{"edgeObj":GraphEdge(source=4,target=2,isCommunicating = 1)}),
         (4,3,{"edgeObj":GraphEdge(source=4,target=3,isCommunicating = 0)}),
         (4,4,{"edgeObj":GraphEdge(source=4,target=4,isCommunicating = 1)})]

# create the whole graph for your system with task and communication edges
MASgraph = nx.Graph()
MASgraph.add_nodes_from(nodes)
MASgraph.add_edges_from(edges)


#########################################################################################################
# Predicate construction
#########################################################################################################



formula   = STLformula(temporalOperator   = "always",
                    predicate             = ellipsoidPredicate(center=np.array([5,10]),P = np.eye(2)/9),
                    timeinterval          = timeInterval(20.,25.))

formula.predicate.addSourceTarget(source=1,target=3)
MASgraph.edges[1,3]["edgeObj"].addFormula(formula)


formula   = STLformula(temporalOperator   = "always",
                        predicate          = ellipsoidPredicate(center=np.array([10,0]),P = np.eye(2)/9),
                        timeinterval       = timeInterval(20.,25.))

formula.predicate.addSourceTarget(source=3,target=4)
MASgraph.edges[4,3]["edgeObj"].addFormula(formula)
    
formula   = STLformula(temporalOperator   = "always",
                        predicate          = ellipsoidPredicate(center=np.array([10,0]),P = np.eye(2)/9),
                        timeinterval       = timeInterval(20.,25.))

formula.predicate.addSourceTarget(source=1,target=2)
MASgraph.edges[1,2]["edgeObj"].addFormula(formula)


formula   = STLformula(temporalOperator   = "eventually",
                        predicate          = ellipsoidPredicate(center=np.array([10,10]),P = np.eye(2)/3),
                        timeinterval       = timeInterval(30.,35.),
                        timeOfSatisfaction =35)

formula.predicate.addSourceTarget(source=4,target=4)
MASgraph.edges[4,4]["edgeObj"].addFormula(formula)

formula   = STLformula(temporalOperator   = "eventually",
                        predicate          = ellipsoidPredicate(center=np.array([0,0]),P = np.eye(2)/9),
                        timeinterval       = timeInterval(30.,35.),
                        timeOfSatisfaction=35)

formula.predicate.addSourceTarget(source=4,target=3)
MASgraph.edges[4,3]["edgeObj"].addFormula(formula)


    
initialTaskGraph,finalTaskGraph,commGraph  = computeNewTaskGraph(MASgraph=MASgraph,problemDimension=2,maxDistanceFunction=maxDistanceFunction)
visualizeGraphs(commGraph, initialTaskGraph, finalTaskGraph)
sys.stdout = orig_stdout
f.close()

initialAgentsState ={1:np.array([-10,-15]),
                     2:np.array([0,5]),
                     3:np.array([0,10]),
                     4:np.array([5,-10])}

stateTrajectories = simulateAgents(finalTaskGraph,tEnd=38,tStart=0,initialAgentsState=initialAgentsState,cleaningTimes=[26])

try:
    os.mkdir("./simulation1")
except OSError as error:
    print(error) 

for agentIndex,trajectory in stateTrajectories.items() :       
        with open(f"simulation1/agent{agentIndex}.npy", 'wb') as f:
                np.save(f,trajectory)

        
plt.show()