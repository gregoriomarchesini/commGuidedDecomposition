# test the controller 
from multiagent_STLdec.control_module import STLController,Agent
from multiagent_STLdec.predicate_builder_module import *
import matplotlib.pylab as plt




problemDimension  = 2
radius            = 2
initialAgentState1 = np.array([0,0])
initialAgentState2 = np.array([20,0])


predicate   = ellipsoidPredicate(np.eye(problemDimension)/radius**2,center=np.array([0,10]))
predicate.addSourceTarget(source=1,target=2)
edgeFormula = STLformula(temporalOperator = "always",
                                timeinterval     = timeInterval(5,10),
                                predicate        = predicate)

predicate    = ellipsoidPredicate(np.eye(problemDimension)/radius**2,center=np.array([10,20]))
predicate.addSourceTarget(source=1,target=2)

edgeFormula2 = STLformula(temporalOperator = "always",
                          timeinterval     = timeInterval(20,25),
                          predicate        = predicate)


agentsIndexList = [1,2]
Agent1 = Agent(initialState=initialAgentState1,agentIndex=1,neigbours=[2]);Agent1.initializeController(initialNeigboursState={2:initialAgentState2},formulas=[edgeFormula,edgeFormula2])
Agent2 = Agent(initialState=initialAgentState2,agentIndex=2,neigbours=[1]);Agent2.initializeController(initialNeigboursState={1:initialAgentState1},formulas=[edgeFormula,edgeFormula2])




agent1States  = []
agent2States  = []
statesTrajectory = {1:agent1States,2:agent2States} 

for t in np.arange(0,25,Agent1.timeStep) :
    # take a step
    agent1NextState = Agent1.step(time=t)
    agent2NextState = Agent2.step(time=t)
    
    Agent1.printCurrentConstraintValue()
    
    statesTrajectory[1].append(agent1NextState)
    statesTrajectory[2].append(agent2NextState)
    
    Agent1.updateNeighboursState(neigboursState={2:agent2NextState})
    Agent2.updateNeighboursState(neigboursState={1:agent1NextState})
    
    


fig,ax = plt.subplots()

for agentId,trajectory in statesTrajectory.items() :
    x = []
    y = []

    for state in trajectory :
        x.append(np.squeeze(state[0]))
        y.append(np.squeeze(state[1]))
    
    if agentId ==1 :
        ax.plot(x,y,c="red")
        ax.scatter(x[0],y[0],c="red")
    else :
        ax.plot(x,y,c="blue")
        ax.scatter(x[0],y[0],c="blue")
            
plt.show()