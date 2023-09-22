import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import networkx as nx
from   multiagent_STLdec.decomposition_module import *

# define optimization problem 


#########################################################################################################
# Graph creation
#########################################################################################################
# create a set of node positions 

v0 = np.array([-5.,-5.])
v1 = np.array([ 5.,-5.])
v2 = np.array([-5.,0.])
v3 = np.array([ 5.,0.])
v4 = np.array([-5.,5.])
v5 = np.array([ 5.,5.])
v6 = np.array([-10.,7.])
v7 = np.array([ 10.,7.])

# select maximum communication distance
maximumInterRobotDistance  = 50                  # not binding for now
problemDimension           = len(v1)             # dimension of the problem in R^n
numberOfVerticesHypercube  = 2**problemDimension # number of vertices in a hypercube in this dimensional space
distPred :convexPredicateFunction     = maxDistancePredicate(stateSpaceDimension=problemDimension,maxDistance=maximumInterRobotDistance) # here the target and source do not matter really
maxDistanceFunction = distPred.function
# create an iterable with index and attributes of the nodes you want to create

nodes =[(0,{"pos":v0}),
        (1,{"pos":v1}),
        (2,{"pos":v2}),
        (3,{"pos":v3}),
        (4,{"pos":v4}),
        (5,{"pos":v5}),
        (6,{"pos":v6}),
        (7,{"pos":v7})]

edges = [(0,2,{"edgeObj":GraphEdge(source=0,target=2,isCommunicating = 1)}),
         (2,4,{"edgeObj":GraphEdge(source=2,target=4,isCommunicating = 1)}),
         (4,6,{"edgeObj":GraphEdge(source=4,target=6,isCommunicating = 1)}),
         (1,3,{"edgeObj":GraphEdge(source=1,target=3,isCommunicating = 1)}),
         (3,5,{"edgeObj":GraphEdge(source=3,target=5,isCommunicating = 1)}),
         (2,3,{"edgeObj":GraphEdge(source=2,target=3,isCommunicating = 0)}),
         (4,5,{"edgeObj":GraphEdge(source=4,target=5,isCommunicating = 1)}),
         (5,7,{"edgeObj":GraphEdge(source=5,target=7,isCommunicating = 1)}),
         (1,7,{"edgeObj":GraphEdge(source=1,target=7,isCommunicating = 1)}),
         (6,0,{"edgeObj":GraphEdge(source=6,target=0,isCommunicating = 1)}),]

# create the whole graph for your system with task and communication edges
MASgraph = nx.Graph()
MASgraph.add_nodes_from(nodes)
MASgraph.add_edges_from(edges)


#########################################################################################################
# Predicate construction
#########################################################################################################

## predicates for communicating agents
# set isotropic distance predicate between agent 23 and 45
radius    = 20
predicate23 = ellipsoidPredicate(np.eye(problemDimension)/radius**2,np.zeros((problemDimension,1)))
predicate45 = ellipsoidPredicate(np.eye(problemDimension)/radius**2,np.zeros((problemDimension,1)))



#TODO: The cycle constraints hgave to be checked before the overloading constraints. This is because during 
# the cycle constraint definition you will have to change some predicates if necessary to close the cycle

formula23   = STLformula(temporalOperator    = "always",
                         predicate           = predicate23,
                         timeinterval        = timeInterval(0,10))

formula45   = STLformula(temporalOperator    = "always",
                         predicate           = predicate45,
                         timeinterval        = timeInterval(0,10))

MASgraph.edges[2,3]["edgeObj"].addFormula(formula23)
MASgraph.edges[4,5]["edgeObj"].addFormula(formula45)


## predicates for non-communicating agents

P28,B28 = computeEllipseMatrix(semiMajorAxis=8,semiMinorAxis=6,theta= 30*np.pi/180)
P17,B28 = computeEllipseMatrix(semiMajorAxis=8,semiMinorAxis=6,theta=-30*np.pi/180)

formula60   = STLformula(temporalOperator    = "always",
                         predicate           = ellipsoidPredicate(P17,v6-v0),
                         timeinterval        = timeInterval(0,10))

formula71   = STLformula(temporalOperator    = "always",
                         predicate           = ellipsoidPredicate(P28,v7-v1),
                         timeinterval        = timeInterval(0,10),)

formula71eventually  = STLformula(temporalOperator   = "eventually",
                                 predicate           = ellipsoidPredicate(P28,v7*2-v1),
                                 timeinterval        = timeInterval(0,10))

MASgraph.edges[6,0]["edgeObj"].addFormula(formula60)
MASgraph.edges[7,1]["edgeObj"].addFormula([formula71,formula71eventually])



initialTaskGraph,TaskGraph,commGraph  = computeNewTaskGraph(MASgraph=MASgraph,problemDimension=2,maxDistanceFunction=maxDistanceFunction)
visualizeGraphs(commGraph, initialTaskGraph, TaskGraph)