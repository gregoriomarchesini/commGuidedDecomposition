import numpy as np
import sys,os
from  networkx import Graph
from   multiagent_STLdec.decomposition_module import *
from  multiagent_STLdec.predicate_builder_module import *
from multiagent_STLdec.control_module import *
from multiagent_STLdec.visualization_module import simulateAgents


orig_stdout = sys.stdout
f = open('simulation1_decompositon.txt', 'w')
sys.stdout = f

# define optimization problem 

#########################################################################################################
# Graph creation
#########################################################################################################
# create a set of node positions 

v1 = np.array([ -20.,-20.])
v2 = np.array([-20.,-10.])
v3 = np.array([ -20.,-5.])
v4 = np.array([-20.,-0.])
v5 = np.array([ 0.,0.])

centerDiamond        = ((v1+v2+v3+v4)-v5)/4 
v6toDiamondDirection = centerDiamond-v5
v6toDiamondDirection = v6toDiamondDirection/np.linalg.norm(v6toDiamondDirection)
midPointOfSeparation = (centerDiamond-v5)/2


# select maximum communication distance
maximumInterRobotDistance  = 40               # not binding for now
problemDimension           = len(v1)           # dimension of the problem in R^n

# create an iterable with index and attributes of the nodes you want to create

nodes =[(1,{"pos":v1}),
        (2,{"pos":v2}),
        (3,{"pos":v3}),
        (4,{"pos":v4}),
        (5,{"pos":v5}),]

edges = [(4,5,{"edgeObj":GraphEdge(source=5,target=4,isCommunicating = 1)}),
         (4,3,{"edgeObj":GraphEdge(source=4,target=3,isCommunicating = 1)}),
         (3,5,{"edgeObj":GraphEdge(source=3,target=5,isCommunicating = 0)}),
         (1,2,{"edgeObj":GraphEdge(source=1,target=2,isCommunicating = 1)}),
         (3,2,{"edgeObj":GraphEdge(source=2,target=3,isCommunicating = 1)}),
         (2,1,{"edgeObj":GraphEdge(source=1,target=2,isCommunicating = 1)}),
         (5,2,{"edgeObj":GraphEdge(source=5,target=2,isCommunicating = 0)}),
         (1,5,{"edgeObj":GraphEdge(source=1,target=5,isCommunicating = 0)}),
         (1,1,{"edgeObj":GraphEdge(source=1,target=1,isCommunicating = 1)}),
         (5,5,{"edgeObj":GraphEdge(source=5,target=5,isCommunicating = 1)}),
         (3,1,{"edgeObj":GraphEdge(source=3,target=1,isCommunicating = 0)}),]

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


edgesToTheDiamond = [(5,1),(5,2),(5,3)]
P,B = computeEllipseMatrix(semiMajorAxis=14,semiMinorAxis=3,theta=np.pi/2)

# first formation task
for i,j in edgesToTheDiamond :
    formula   = STLformula(temporalOperator   = "always",
                           predicate          = ellipsoidPredicate(center=np.array([-20,0]),P = P),
                           timeinterval       = timeInterval(20.,24.))
    formula.predicate.addSourceTarget(source=i,target=j)
    MASgraph.edges[i,j]["edgeObj"].addFormula(formula)


# self task of 5
formula   = STLformula(temporalOperator   = "always",
                           predicate      = ellipsoidPredicate(center=np.array([0,0]),P = np.eye(2)/9),
                           timeinterval   = timeInterval(20.,24.))
formula.predicate.addSourceTarget(source=5,target=5)
MASgraph.edges[5,5]["edgeObj"].addFormula(formula)

# self task of 

formula   = STLformula(temporalOperator   = "eventually",
                           predicate      = ellipsoidPredicate(center=np.array([-25,0]),P = np.eye(2)/4),
                           timeinterval   = timeInterval(29.,31.),
                           timeOfSatisfaction=30)
formula.predicate.addSourceTarget(source=1,target=1)
MASgraph.edges[1,1]["edgeObj"].addFormula(formula)

formula   = STLformula(temporalOperator   = "always",
                           predicate      = ellipsoidPredicate(center=np.array([0,8]),P = np.eye(2)/4),
                           timeinterval   = timeInterval(29.,31.))
formula.predicate.addSourceTarget(source=3,target=1)
MASgraph.edges[3,1]["edgeObj"].addFormula(formula)




initialTaskGraph,finalTaskGraph,commGraph  = computeNewTaskGraph(MASgraph=MASgraph,problemDimension=2,maxInterRobotDistance= maximumInterRobotDistance)
visualizeGraphs(commGraph, initialTaskGraph, finalTaskGraph)
sys.stdout = orig_stdout
f.close()


initialAgentsState ={ 1:np.array([0,0]),
                      2:np.array([-10,11]),
                      3:np.array([-25,-30]),
                      4:np.array([0,-20]),
                      5:np.array([15,-15])}

stateTrajectories = simulateAgents(finalTaskGraph,tEnd=32,tStart=0,initialAgentsState=initialAgentsState,cleaningTimes=[25])


try:
    os.mkdir("./simulation1")
except OSError as error:
    print(error) 

for agentIndex,trajectory in stateTrajectories.items() :       
        with open(f"simulation1/agent{agentIndex}.npy", 'wb') as f:
                np.save(f,trajectory)

        
plt.show()

    
    
    