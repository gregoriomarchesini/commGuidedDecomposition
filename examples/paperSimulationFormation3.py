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

nodes ={0:v1,
        1:v2,
        2:v3,
        3:v4,
        4:v5,
        5:v6,
        6:v7,
        7:v8}


communicatingEdges = [(0,1),(0,2),(0,4),(0,3),(4,5),(5,3),(5,6),(5,7),(5,5),(1,1),(0,0)]
getCompleteGraph(n=8,communicatingEdges=communicatingEdges,nodePositions=nodes)






# xx = [node[0]["pos"][0] for node in nodes]
# yy = [node[0]["pos"][1] for node in nodes]
# xxmin,xxmax = min(xx)*1.6,max(xx)*1.6
# yymin,yymax = min(yy)*1.6,max(yy)*1.6

# # create the whole graph for your system with task and communication edges
# MASgraph = Graph()
# MASgraph.add_nodes_from(nodes)
# MASgraph.add_edges_from(edges)


# #########################################################################################################
# # Predicate construction
# #########################################################################################################

# # formation task
# formula   = STLformula(temporalOperator   = "always",
#                            predicate      = ellipsoidPredicate(center=np.array([-15,15]),P = np.eye(2)/9),
#                            timeinterval   = timeInterval(10.,15.))
# formula.predicate.addSourceTarget(source=4,target=7) # if you don't specify it , it will take the source,target from the edge
# MASgraph.edges[7,4]["edgeObj"].addFormula(formula)

# formula   = STLformula(temporalOperator   = "always",
#                            predicate      = ellipsoidPredicate(center=np.array([-15,-15]),P = np.eye(2)/9),
#                            timeinterval   = timeInterval(10.,15.))
# formula.predicate.addSourceTarget(source=4,target=1)
# MASgraph.edges[1,4]["edgeObj"].addFormula(formula)

# formula   = STLformula(temporalOperator   = "always",
#                            predicate      = ellipsoidPredicate(center=np.array([15,-15]),P = np.eye(2)/9),
#                            timeinterval   = timeInterval(10.,15.))
# formula.predicate.addSourceTarget(source=3,target=2)
# MASgraph.edges[3,2]["edgeObj"].addFormula(formula)

# formula   = STLformula(temporalOperator   = "always",
#                            predicate      = ellipsoidPredicate(center=np.array([15,15]),P = np.eye(2)/9),
#                            timeinterval   = timeInterval(10.,15.))
# formula.predicate.addSourceTarget(source=3,target=6)
# MASgraph.edges[3,6]["edgeObj"].addFormula(formula)



# formula   = STLformula(temporalOperator   = "always",
#                            predicate      = ellipsoidPredicate(center=np.array([10,-10]),P = np.eye(2)/4),
#                            timeinterval   = timeInterval(10.,15.))
# formula.predicate.addSourceTarget(source=5,target=3)
# MASgraph.edges[5,3]["edgeObj"].addFormula(formula)



# formula   = STLformula(temporalOperator   = "always",
#                            predicate      = ellipsoidPredicate(center=np.array([-10,-10]),P = np.eye(2)/4),
#                            timeinterval   = timeInterval(10.,15.))
# formula.predicate.addSourceTarget(source=5,target=4)
# MASgraph.edges[5,4]["edgeObj"].addFormula(formula)

# # Single agent task for 5

# formula   = STLformula(temporalOperator   = "eventually",
#                            predicate      = polytopicSetPredicate(center=np.array([0,0]),a_list=[np.array([0,-1])],distances=[0],computeApprox=False),
#                            timeinterval   = timeInterval(10.,15.),
#                            timeOfSatisfaction = 15)
# formula.predicate.addSourceTarget(source=5,target=5)
# MASgraph.edges[5,5]["edgeObj"].addFormula(formula)



# formula   = STLformula(temporalOperator   = "eventually",
#                            predicate      = polytopicSetPredicate(center=np.array([0,0]),a_list=[np.array([0,-1])],distances=[0],computeApprox=False),
#                            timeinterval   = timeInterval(10.,15.),
#                            timeOfSatisfaction = 15)
# formula.predicate.addSourceTarget(source=5,target=5)
# MASgraph.edges[5,5]["edgeObj"].addFormula(formula)



# # Separation Task



# formula   = STLformula(temporalOperator   = "eventually",
#                            predicate      = polytopicSetPredicate(center=np.array([5,0]),a_list=[np.array([-1,0])],distances=[0],computeApprox=False),
#                            timeinterval   = timeInterval(25.,28.),
#                            timeOfSatisfaction=27)
# formula.predicate.addSourceTarget(source=5,target=5)
# MASgraph.edges[5,5]["edgeObj"].addFormula(formula)


# formula   = STLformula(temporalOperator   = "eventually",
#                            predicate      = polytopicSetPredicate(center=np.array([5,0]),a_list=[np.array([-1,0])],distances=[0],computeApprox=False),
#                            timeinterval   = timeInterval(25.,28.),
#                            timeOfSatisfaction = 26)
# formula.predicate.addSourceTarget(source=0,target=0)
# MASgraph.edges[0,0]["edgeObj"].addFormula(formula)

# formula   = STLformula(temporalOperator   = "eventually",
#                            predicate      = polytopicSetPredicate(center=np.array([0,5]),a_list=[np.array([0,-1])],distances=[0],computeApprox=False),
#                            timeinterval   = timeInterval(25.,28.),
#                            timeOfSatisfaction=27)
# formula.predicate.addSourceTarget(source=5,target=5)
# MASgraph.edges[5,5]["edgeObj"].addFormula(formula)


# formula   = STLformula(temporalOperator   = "eventually",
#                            predicate      = polytopicSetPredicate(center=np.array([0,-5]),a_list=[np.array([0,1])],distances=[0],computeApprox=False),
#                            timeinterval   = timeInterval(25.,28.),
#                            timeOfSatisfaction = 26)
# formula.predicate.addSourceTarget(source=0,target=0)
# MASgraph.edges[0,0]["edgeObj"].addFormula(formula)


# formula   = STLformula(temporalOperator   = "always",
#                            predicate      = ellipsoidPredicate(center=np.array([+16,0]),P = np.eye(2)/8),
#                            timeinterval   = timeInterval(25.,28.))
# formula.predicate.addSourceTarget(source=1,target=2)
# MASgraph.edges[1,2]["edgeObj"].addFormula(formula)


# formula   = STLformula(temporalOperator   = "always",
#                            predicate      = ellipsoidPredicate(center=np.array([+16,0]),P = np.eye(2)/8),
#                            timeinterval   = timeInterval(25.,28.))
# formula.predicate.addSourceTarget(source=7,target=6)
# MASgraph.edges[6,7]["edgeObj"].addFormula(formula)



# formula   = STLformula(temporalOperator   = "always",
#                            predicate      = ellipsoidPredicate(center=np.array([-8,-8]),P = np.eye(2)/8),
#                            timeinterval   = timeInterval(25.,28.))
# formula.predicate.addSourceTarget(source=0,target=1)
# MASgraph.edges[0,1]["edgeObj"].addFormula(formula)

# formula   = STLformula(temporalOperator   = "always",
#                            predicate      = ellipsoidPredicate(center=np.array([-8,8]),P = np.eye(2)/8),
#                            timeinterval   = timeInterval(25.,28.))
# formula.predicate.addSourceTarget(source=5,target=7)
# MASgraph.edges[5,7]["edgeObj"].addFormula(formula)





# initialTaskGraph,finalTaskGraph,commGraph  = computeNewTaskGraph(MASgraph=MASgraph,problemDimension=2,maxInterRobotDistance= maximumInterRobotDistance)
# visualizeGraphs(commGraph, initialTaskGraph, finalTaskGraph)
# sys.stdout = orig_stdout
# f.close()


# initialAgentsState ={ 0:np.array([0,0]),
#                       1:np.array([-10,11]),
#                       2:np.array([-25,-30]),
#                       3:np.array([0,20]),
#                       4:np.array([-15,15]),
#                       5:np.array([15,-30]),
#                       6:np.array([-30,-15]),
#                       7:np.array([11,0])}

# stateTrajectories = simulateAgents(finalTaskGraph,tEnd=28,tStart=0,initialAgentsState=initialAgentsState,cleaningTimes=[15.4])


# try:
#     os.mkdir("./simulation3")
# except OSError as error:
#     print(error) 

# for agentIndex,trajectory in stateTrajectories.items() :       
#         with open(f"simulation3/agent{agentIndex}.npy", 'wb') as f:
#                 np.save(f,trajectory)

        
# plt.show()

    
    
    