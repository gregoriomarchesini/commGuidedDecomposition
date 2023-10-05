import numpy as np
import sys,os
from  networkx import Graph
from   multiagent_STLdec.decomposition_module import *
from  multiagent_STLdec.predicate_builder_module import *
from multiagent_STLdec.control_module import *
from multiagent_STLdec.visualization_module import simulateAgents


orig_stdout = sys.stdout
f = open('simulation3_decompositon.txt', 'w')
sys.stdout = f

# define optimization problem 

#########################################################################################################
# Graph creation
#########################################################################################################
# create a set of node positions 

v1 = np.array([ 0.,0])
v2 = np.array([-20.,5.])
v3 = np.array([ 20.,5.])
v4 = np.array([ 5.,8.])
v5 = np.array([ -5.,8.])

v6 = np.array([ 0.,15.])
v7 = np.array([10.,25.])
v8 = np.array([ -10.,25.])


# select maximum communication distance
maximumInterRobotDistance  = 1000               # not binding for now
problemDimension           = len(v1)           # dimension of the problem in R^n

# create an iterable with index and attributes of the nodes you want to create

nodes =[(1,{"pos":v1}),
        (2,{"pos":v2}),
        (3,{"pos":v3}),
        (4,{"pos":v4}),
        (5,{"pos":v5}),
        (6,{"pos":v6}),
        (7,{"pos":v7}),
        (8,{"pos":v8}),]


communicatingEdges = [(1,2),(1,3),(1,5),(1,4),(5,6),(6,4),(6,7),(6,8),(6,6),(2,2),(1,1)]
nonCommunicatingEdges = [(2,5),(4,3),(7,4),(8,5),(8,2),(7,3),(8,7),(3,2)]

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
                           predicate      = ellipsoidPredicate(center=np.array([-15,15]),P = np.eye(2)/9),
                           timeinterval   = timeInterval(10.,15.))
formula.predicate.addSourceTarget(source=5,target=8) # if you don't specify it , it will take the source,target from the edge
MASgraph.edges[8,5]["edgeObj"].addFormula(formula)

formula   = STLformula(temporalOperator   = "always",
                           predicate      = ellipsoidPredicate(center=np.array([-15,-15]),P = np.eye(2)/9),
                           timeinterval   = timeInterval(10.,15.))
formula.predicate.addSourceTarget(source=5,target=2)
MASgraph.edges[2,5]["edgeObj"].addFormula(formula)

formula   = STLformula(temporalOperator   = "always",
                           predicate      = ellipsoidPredicate(center=np.array([15,-15]),P = np.eye(2)/9),
                           timeinterval   = timeInterval(10.,15.))
formula.predicate.addSourceTarget(source=4,target=3)
MASgraph.edges[4,3]["edgeObj"].addFormula(formula)

formula   = STLformula(temporalOperator   = "always",
                           predicate      = ellipsoidPredicate(center=np.array([15,15]),P = np.eye(2)/9),
                           timeinterval   = timeInterval(10.,15.))
formula.predicate.addSourceTarget(source=4,target=7)
MASgraph.edges[4,7]["edgeObj"].addFormula(formula)



formula   = STLformula(temporalOperator   = "always",
                           predicate      = ellipsoidPredicate(center=np.array([10,-10]),P = np.eye(2)/4),
                           timeinterval   = timeInterval(10.,15.))
formula.predicate.addSourceTarget(source=6,target=4)
MASgraph.edges[6,4]["edgeObj"].addFormula(formula)



formula   = STLformula(temporalOperator   = "always",
                           predicate      = ellipsoidPredicate(center=np.array([-10,-10]),P = np.eye(2)/4),
                           timeinterval   = timeInterval(10.,15.))
formula.predicate.addSourceTarget(source=6,target=5)
MASgraph.edges[6,5]["edgeObj"].addFormula(formula)

# Single agent task for 6

formula   = STLformula(temporalOperator   = "eventually",
                           predicate      = polytopicSetPredicate(center=np.array([0,0]),a_list=[np.array([0,-1])],distances=[0],computeApprox=False),
                           timeinterval   = timeInterval(10.,15.),
                           timeOfSatisfaction = 15)
formula.predicate.addSourceTarget(source=6,target=6)
MASgraph.edges[6,6]["edgeObj"].addFormula(formula)



formula   = STLformula(temporalOperator   = "eventually",
                           predicate      = polytopicSetPredicate(center=np.array([0,0]),a_list=[np.array([0,-1])],distances=[0],computeApprox=False),
                           timeinterval   = timeInterval(10.,15.),
                           timeOfSatisfaction = 15)
formula.predicate.addSourceTarget(source=6,target=6)
MASgraph.edges[6,6]["edgeObj"].addFormula(formula)



# Separation Task



formula   = STLformula(temporalOperator   = "eventually",
                           predicate      = polytopicSetPredicate(center=np.array([5,0]),a_list=[np.array([-1,0])],distances=[0],computeApprox=False),
                           timeinterval   = timeInterval(25.,28.),
                           timeOfSatisfaction=27)
formula.predicate.addSourceTarget(source=6,target=6)
MASgraph.edges[6,6]["edgeObj"].addFormula(formula)


formula   = STLformula(temporalOperator   = "eventually",
                           predicate      = polytopicSetPredicate(center=np.array([5,0]),a_list=[np.array([-1,0])],distances=[0],computeApprox=False),
                           timeinterval   = timeInterval(25.,28.),
                           timeOfSatisfaction = 26)
formula.predicate.addSourceTarget(source=1,target=1)
MASgraph.edges[1,1]["edgeObj"].addFormula(formula)

formula   = STLformula(temporalOperator   = "eventually",
                           predicate      = polytopicSetPredicate(center=np.array([0,5]),a_list=[np.array([0,-1])],distances=[0],computeApprox=False),
                           timeinterval   = timeInterval(25.,28.),
                           timeOfSatisfaction=27)
formula.predicate.addSourceTarget(source=6,target=6)
MASgraph.edges[6,6]["edgeObj"].addFormula(formula)


formula   = STLformula(temporalOperator   = "eventually",
                           predicate      = polytopicSetPredicate(center=np.array([0,-5]),a_list=[np.array([0,1])],distances=[0],computeApprox=False),
                           timeinterval   = timeInterval(25.,28.),
                           timeOfSatisfaction = 26)
formula.predicate.addSourceTarget(source=1,target=1)
MASgraph.edges[1,1]["edgeObj"].addFormula(formula)


formula   = STLformula(temporalOperator   = "always",
                           predicate      = ellipsoidPredicate(center=np.array([+16,0]),P = np.eye(2)/8),
                           timeinterval   = timeInterval(25.,28.))
formula.predicate.addSourceTarget(source=2,target=3)
MASgraph.edges[2,3]["edgeObj"].addFormula(formula)


formula   = STLformula(temporalOperator   = "always",
                           predicate      = ellipsoidPredicate(center=np.array([+16,0]),P = np.eye(2)/8),
                           timeinterval   = timeInterval(25.,28.))
formula.predicate.addSourceTarget(source=8,target=7)
MASgraph.edges[7,8]["edgeObj"].addFormula(formula)



formula   = STLformula(temporalOperator   = "always",
                           predicate      = ellipsoidPredicate(center=np.array([-8,-8]),P = np.eye(2)/8),
                           timeinterval   = timeInterval(25.,28.))
formula.predicate.addSourceTarget(source=1,target=2)
MASgraph.edges[1,2]["edgeObj"].addFormula(formula)

formula   = STLformula(temporalOperator   = "always",
                           predicate      = ellipsoidPredicate(center=np.array([-8,8]),P = np.eye(2)/8),
                           timeinterval   = timeInterval(25.,28.))
formula.predicate.addSourceTarget(source=6,target=8)
MASgraph.edges[6,8]["edgeObj"].addFormula(formula)





initialTaskGraph,finalTaskGraph,commGraph  = computeNewTaskGraph(MASgraph=MASgraph,problemDimension=2,maxInterRobotDistance= maximumInterRobotDistance)
visualizeGraphs(commGraph, initialTaskGraph, finalTaskGraph)
sys.stdout = orig_stdout
f.close()


initialAgentsState ={ 1:np.array([0,0]),
                      2:np.array([-10,11]),
                      3:np.array([-25,-30]),
                      4:np.array([0,20]),
                      5:np.array([-15,15]),
                      6:np.array([15,-30]),
                      7:np.array([-30,-15]),
                      8:np.array([11,0])}

stateTrajectories = simulateAgents(finalTaskGraph,tEnd=28,tStart=0,initialAgentsState=initialAgentsState,cleaningTimes=[15.4])


try:
    os.mkdir("./simulation3")
except OSError as error:
    print(error) 

for agentIndex,trajectory in stateTrajectories.items() :       
        with open(f"simulation3/agent{agentIndex}.npy", 'wb') as f:
                np.save(f,trajectory)

        
plt.show()

    
    
    