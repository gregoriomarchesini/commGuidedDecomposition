import matplotlib.pyplot as plt
from .control_module import Agent
import numpy as np
from tqdm import tqdm
from networkx import Graph
plt.rcParams["figure.figsize"] = (3.425, 2.325)


def simulateAgents(taskGraph:Graph,tStart:float,tEnd:float,initialAgentsState:dict[int,np.ndarray],relevantInstants:list =[], cleaningTimes:list = []):
    
    agents            :dict[int,Agent]= {} # Agents obj
    agentsState       :dict[int,np.ndarray]= {} # save the current agents states
    agentsTrajectory  :dict[int,list[np.ndarray]]= {}
    
    if len(cleaningTimes)==0 :
        cleaningTimes = [1E20]
        
    # time to create a very cool simulation 
    savedTimeStep = 0
    for agentID,initialState in initialAgentsState.items() :
        neigbours = list(taskGraph.neighbors(agentID))
        
        agentsState[agentID] = initialState
        agentsTrajectory[agentID] = [initialState]
        
        agents[agentID] = Agent(initialState= initialState  ,agentIndex=agentID,neigbours=neigbours)
        if not savedTimeStep: #save time step
            deltaT = agents[agentID].timeStep
            savedTimeStep=1
        
    formulasForAgent = {index: [] for index in initialAgentsState.keys()}
    for source,target,dataObj in taskGraph.edges(data=True) :
        edgeObj = dataObj["edgeObj"]
        if isinstance(edgeObj.formulasList,list) :
            formulasForAgent[source] += edgeObj.formulasList
            formulasForAgent[target] += edgeObj.formulasList
        else :
            formulasForAgent[source] += [edgeObj.formulasList]
            formulasForAgent[target] += [edgeObj.formulasList]
    
    print("--------------------------------")
    print("Constraint per agent report")
    print("--------------------------------")
    # initialise
    for agentId,agent in agents.items() :
        # here target source is just a name. Doesn't mean that the edge has this direction. The direction of the edge is found later on inside the code
        neigboursState = {}
        for index in agents[agentId].neigbours :
            neigboursState[index] = agentsState[index]
        print(f"agentID : {agentId}")
        print(f"number of constraints : {len(formulasForAgent[agentId])}")
        agent.initializeController(formulas               = formulasForAgent[agentId],
                                   initialNeigboursState  = neigboursState,
                                   initializationTime     = tStart,
                                   allowSlackSatisfaction = True)
    print("---------------------------------")
    print("Note that each polytopic constraint amounts to a number of constraints equal to the number of edges")
    timeRange = np.arange(tStart,tEnd,deltaT)
    barrierValues = []
    barrierConstraintValues = []

    maxIt   = len(timeRange)
    relevantInstantsIndex = []
    for instant in relevantInstants :
        relevantInstantsIndex.append(int((instant-tStart)/deltaT)) # change the to an index
        if instant <= 0 or instant>=maxIt :
            print("One or more time instants is outside the time bouds of the simulation")
        
    
    
    closestCleaningTime = cleaningTimes.pop(0)
    for t in tqdm(timeRange[:-1]) :
        currentBarrierValue = []
        currentConstraintValues = []
        
        if t >= closestCleaningTime : # reinitialize controller
            print(f"reinitializing controller at time : {t}")
            for agentId,agent in agents.items() :
                # here target source is just a name. Doesn't mean that the edge has this direction. The direction of the edge is found later on inside the code
                neigboursState = {}
                for index in agents[agentId].neigbours :
                    neigboursState[index] = agentsState[index]
                agent.cleanController(initialNeigboursState =neigboursState,
                                      initializationTime = t,
                                      allowSlackSatisfaction = True) 
                if len(cleaningTimes) :
                    closestCleaningTime = cleaningTimes.pop(0)
                else :
                    closestCleaningTime= 10E20 # just a big number
        
       
        for agentIndex,agent in agents.items() :
            # take a step
            agentNextState          = np.squeeze(agent.step(time=t)) # to remove extradimension
            agentsState[agentIndex] = agentNextState
            agentsTrajectory[agentIndex].append(agentNextState)
            currentBarrierValue += agent.getListOfBarrierValuesAtCurrentTime()
            currentConstraintValues += agent.getListOfBarrierConstraintValuesAtCurrentTime()
        
        barrierValues.append(currentBarrierValue)
        barrierConstraintValues.append(currentConstraintValues)
        if t==timeRange[0] :
            intialNumberOfBarriers = len(currentBarrierValue)
            
        # now that all the states are updated we can the compute the next control input
        for agentIndex,agent in agents.items() :
            neigboursState = {}
            for index in agents[agentIndex].neigbours :
                neigboursState[index] = agentsState[index]
                agent.updateNeighboursState(neigboursState = neigboursState)
        

    fig,ax   = plt.subplots()
    fig2,ax2 = plt.subplots(1,2)
    fig3,ax3 = plt.subplots()
    
    decimation = 1/100 # between 0 and 1 (percentage of total number of points to plot)
    
    ax.grid(visible=True)
    
    for key,trajectory in agentsTrajectory.items() :
        agentsTrajectory[key] = np.hstack((np.stack(trajectory,axis=0),timeRange[:,np.newaxis]))
        
    
    print("Initial Agents State")
    for agentId,trajectory in agentsTrajectory.items() :
        
        x = trajectory[:,0]
        y = trajectory[:,1]

        step  = int(len(x)*decimation)
        color = np.ones((1,4))
        color[0,:3] = np.random.random(3)
        C = np.repeat(color, len(x[::step ]), axis=0)
        C[:,3] =C[:,3]*np.linspace(0.3,1.,len(x[::step ]))
        
        ax.scatter(x[::step],y[::step],c = C)
        ax.scatter(x[0],y[0],c="red", linewidths=3)
        ax.scatter(x[-1],y[-1],c="green",marker="x", linewidths=5)
        
        for timeInstance,indexTimeInstance in zip(relevantInstants,relevantInstantsIndex) :
            ax.scatter(x[indexTimeInstance],y[indexTimeInstance],c="blue",marker="x", linewidths=5)
            ax.annotate(xy=(x[indexTimeInstance]+0.2,y[indexTimeInstance]+0.2),text=f"t = {timeInstance}")
            
        
        ax3.plot(x,y)
        ax3.scatter(x[0],y[0],c="red", linewidths=3)
        ax3.scatter(x[-1],y[-1],c="green",marker="x", linewidths=5)
        
        
        
        ax3.annotate(xy=(x[0]+0.6,y[0]+0.6),text=f"agent {agentId}")
        ax.annotate(xy=(x[0]+0.6,y[0]+0.6),text=f"agent {agentId}")
        print(f"Agent ID : {agentId}: {x[0]},{y[0]}")

    
    ax.set_xlabel("x-axis [m]")
    ax.set_ylabel("x-axis [m]")
    ax.set_title("simulated agents trajectories")
    
    
    for barrier,constraint in zip(barrierValues, barrierConstraintValues) :
        
        if len(barrier) < intialNumberOfBarriers :
            barrier    += [float(0.),]*(intialNumberOfBarriers-len(barrier))
            constraint += [float(0.),]*(intialNumberOfBarriers-len(constraint))

   
    barrierValues = np.array(barrierValues)
    barrierConstraintValues = np.array(barrierConstraintValues)
    for kk in range(len(barrierValues[0,:])) :
        ax2[0].plot(timeRange[:-1],barrierValues[:,kk],scaley="log")
        ax2[1].plot(timeRange[:-1],barrierConstraintValues[:,kk],scaley="log")
        
    ax2[0].set_ylabel("b(x,t)")
    ax2[0].set_xlabel("time [s]")
    ax2[0].set_title("barrier functions time evolution")
    
    ax2[1].set_ylabel("db(x,t)/dt")
    ax2[1].set_xlabel("time [s]")
    ax2[1].set_title("barrier constraint functions time evolution")
    
    
    return agentsTrajectory
    