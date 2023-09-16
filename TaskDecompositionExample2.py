import numpy as np
import casadi as ca
import networkx as nx
from   decomposition_module import *
from  predicate_builder_module import *
from control_module import *
import sys

orig_stdout = sys.stdout
f = open('out.txt', 'w')
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

edges = [(5,6,{"edgeObj":GraphEdge(source=5,target=6,isCommunicating = 1)}),
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
                           predicate          = ellipsoidPredicate(center=np.array([-10,-5]),P = np.eye(2)/4),
                           timeinterval       = timeInterval(20.,40.))
   
    
    MASgraph.edges[i,j]["edgeObj"].addFormula(formula)



initialTaskGraph,finalTaskGraph,commGraph  = computeNewTaskGraph(MASgraph=MASgraph,problemDimension=2,maxDistanceFunction=maxDistanceFunction)
visualizeGraphs(commGraph, initialTaskGraph, finalTaskGraph)

sys.stdout = orig_stdout
f.close()

agents            :dict[int,Agent]= {} # Agents obj
agentsState       :dict[int,np.ndarray]= {} # save the current agents states
agentsTrajectory  :dict[int,list[np.ndarray]]= {}

# time to create a very cool simulation 
for index,posDict in nodes :
    neigbours = list(finalTaskGraph.neighbors(index))
    ss = np.shape((posDict["pos"]))
    initialAgentState = posDict["pos"] + np.random.rand(*ss)*40
    agentsState[index] = initialAgentState 
    agentsTrajectory[index] = [initialAgentState ]
    
    agents[index] = Agent(initialState= initialAgentState  ,agentIndex=index,neigbours=neigbours)
    
formulasForAgent = {index: [] for index,data in nodes}
for source,target,dataObj in finalTaskGraph.edges(data=True) :
    edgeObj :GraphEdge = dataObj["edgeObj"]
    if isinstance(edgeObj.formulasList,list) :
        formulasForAgent[source] += edgeObj.formulasList
        formulasForAgent[target] += edgeObj.formulasList
    else :
        formulasForAgent[source] += [edgeObj.formulasList]
        formulasForAgent[target] += [edgeObj.formulasList]

# initialise
for agentId,agent in agents.items() :
    # here target source is just a name. Doesn't mean that the edge has this direction. The direction of the edge is found later on inside the code
    neigboursState = {}
    for index in agents[agentId].neigbours :
        neigboursState[index] = agentsState[index]
    print(f"agentID : {agentId}")
    print(f"number of constraints : {len(formulasForAgent[agentId])}")
    agent.initializeController(formulas = formulasForAgent[agentId],initialNeigboursState=neigboursState,allowSlackSatisfaction=True)
    
counter = 0
timeRange = np.arange(0,40,agents[1].timeStep)
maxIt   = len(timeRange)
for t in timeRange :
    print(counter/maxIt)

    for agentIndex,agent in agents.items() :
        # take a step
        agentNextState          = agent.step(time=t)
        agentsState[agentIndex] = agentNextState
        agentsTrajectory[agentIndex].append(agentNextState)
        
        # agent.printCurrentConstraintValue()
    
    # now that all the states are updated we can the compute the next control input
    for agentIndex,agent in agents.items() :
        neigboursState = {}
        for index in agents[agentIndex].neigbours :
            neigboursState[index] = agentsState[index]
            agent.updateNeighboursState(neigboursState = neigboursState)

    counter +=1

fig,ax = plt.subplots()

for agentId,trajectory in agentsTrajectory.items() :
    x = []
    y = []

    for state in trajectory :
        x.append(np.squeeze(state[0]))
        y.append(np.squeeze(state[1]))
    
    
    ax.plot(x,y,c="red")
    ax.plot(x[0::int(20/maxIt)*maxIt],y[0::int(20/maxIt)*maxIt],marker=">",c="red")
    ax.scatter(x[0],y[0],c="red")
    ax.scatter(x[-1],y[-1],c="k",marker="+")
    
  
    
    ax.annotate(xy=(x[0],y[0]),text=f"agent {agentId}")
           
plt.show()
    
    
    
    
    