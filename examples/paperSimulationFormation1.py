import numpy as np
import casadi as ca
import networkx as nx
from   multiagent_STLdec.decomposition_module import *
from  multiagent_STLdec.predicate_builder_module import *
from multiagent_STLdec.control_module import *
from multiagent_STLdec.visualization_module import simulateAgents
import sys


orig_stdout = sys.stdout
f = open('simulation1_decompositon.txt', 'w')
sys.stdout = f

# define optimization problem 

#########################################################################################################
# Graph creation
#########################################################################################################
# create a set of node positions 

v1 = np.array([-0.,-4.])
v2 = np.array([ 4.,-0.])
v3 = np.array([-0.,4.])
v4 = np.array([ -4.,0.])
v5 = np.array([2.,10.])
v6 = np.array([ 20.,10.])

centerDiamond        = ((v1+v2+v3+v4)-v6)/4 
v6toDiamondDirection = centerDiamond-v6
v6toDiamondDirection = v6toDiamondDirection/np.linalg.norm(v6toDiamondDirection)
midPointOfSeparation = centerDiamond-v6/2


# select maximum communication distance
maximumInterRobotDistance  = 2000                  # not binding for now
problemDimension           = len(v1)             # dimension of the problem in R^n
distPred :convexPredicateFunction     = maxDistancePredicate(stateSpaceDimension=problemDimension,maxDistance=maximumInterRobotDistance) # here the target and source do not matter really
maxDistanceFunction = distPred.function

# create an iterable with index and attributes of the nodes you want to create

nodes =[(1,{"pos":v1}),
        (2,{"pos":v2}),
        (3,{"pos":v3}),
        (4,{"pos":v4}),
        (5,{"pos":v5}),
        (6,{"pos":v6})]

edges = [(5,6,{"edgeObj":GraphEdge(source=6,target=5,isCommunicating = 1)}),
         (5,3,{"edgeObj":GraphEdge(source=5,target=3,isCommunicating = 1)}),
         (3,4,{"edgeObj":GraphEdge(source=3,target=4,isCommunicating = 1)}),
         (4,1,{"edgeObj":GraphEdge(source=4,target=1,isCommunicating = 1)}),
         (1,2,{"edgeObj":GraphEdge(source=1,target=2,isCommunicating = 1)}),
         (6,3,{"edgeObj":GraphEdge(source=6,target=3,isCommunicating = 0)}),
         (6,2,{"edgeObj":GraphEdge(source=6,target=2,isCommunicating = 0)}),
         (6,1,{"edgeObj":GraphEdge(source=6,target=1,isCommunicating = 0)}),
         (6,4,{"edgeObj":GraphEdge(source=6,target=4,isCommunicating = 0)}),]

xx = [node[1]["pos"][0] for node in nodes]
yy = [node[1]["pos"][1] for node in nodes]
xxmin,xxmax = min(xx)*1.6,max(xx)*1.6
yymin,yymax = min(yy)*1.6,max(yy)*1.6

# create the whole graph for your system with task and communication edges
MASgraph = nx.Graph()
MASgraph.add_nodes_from(nodes)
MASgraph.add_edges_from(edges)


#########################################################################################################
# Predicate construction
#########################################################################################################


edgesToTheDiamond = [(6,1),(6,2),(6,3),(6,4)]



for i,j in edgesToTheDiamond :
    # formula   = STLformula(temporalOperator   = "always",
    #                        predicate          = polytopicSetPredicate(center=midPointOfSeparation,a_list=[-4*v6toDiamondDirection],distances=[0]),
    #                        timeinterval       = timeInterval(20.,40.))
    
    formula   = STLformula(temporalOperator   = "always",
                           predicate          = ellipsoidPredicate(center=np.array([-40,0]),P = np.eye(2)/6),
                           timeinterval       = timeInterval(20.,40.))
   
    
    MASgraph.edges[i,j]["edgeObj"].addFormula(formula)



initialTaskGraph,finalTaskGraph,commGraph  = computeNewTaskGraph(MASgraph=MASgraph,problemDimension=2,maxDistanceFunction=maxDistanceFunction)
visualizeGraphs(commGraph, initialTaskGraph, finalTaskGraph)
sys.stdout = orig_stdout
f.close()


initialAgentsState ={1:np.array([0,0]),
                     2:np.array([-30,-60]),
                     3:np.array([10,11]),
                     4:np.array([-15,-30]),
                     5:np.array([-10,-40]),
                     6:np.array([15,-15])}

simulateAgents(finalTaskGraph,tEnd=40,tStart=0,initialAgentsState=initialAgentsState)

plt.show()
    
    
    
    
    