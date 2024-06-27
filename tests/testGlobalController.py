from stldec.decomposition_module import *
from stldec.predicate_builder_module import *
import casadi as ca




# TODO: add interface for more complex dynamics compared to a single 2D integrator
class STLControllerLogExp():
    """STL controller class"""
    def __init__(self,agentIndex : int,globalAgentsIndices:list[int],stateSpaceDim:int) -> None:
        
        self._STLformulas        : list[STLformula] =  [] # list of all the STL formulas in the system 
        self._agentDynamics      : ca.Function      = None 
        self._agentID         : int                 = agentIndex
        self._opti               : ca.Opti          = ca.Opti()
        self._agentsIndices      : list[int]        = globalAgentsIndices
        
        self._stateSpaceDim      : int        = stateSpaceDim
        self._barrierConstraints :list        = []         # barrier constraint
        self._barrierGradients   :list        =[]
        self._barrierValues      : list       = []
        self._activationFunctions :list       = []
        
        
        self._previousControlInput            = None
        self._previousState                   = None
        
        self._previouslyInitalised = False
        
        
        # variables for function creation
        self._timeVar                     = ca.MX.sym("dd",1)
        self._allAgentsStakedStateVarDict = {agentID:ca.MX.sym("xj",self._stateSpaceDim ) for agentID in self._agentsIndices} # state variable for the neighbour}
        
        self._agentStateVar               = self._allAgentsStakedStateVarDict[self._agentID] # current agent state
        
        self._allAgentsStakedStateVar     = ca.vertcat(*[self._allAgentsStakedStateVarDict[agentID] for agentID in self._agentsIndices]) # stacked version of the dictionary
        self._scaleFactorVar              = ca.MX.sym("scale",1) # it will serve to put the minimum approximation at a postive value
        
        
        # set up all the parameters for optimization
        self._allAgentsStakedStateParDict = {agentID:self._opti.parameter(self._stateSpaceDim ,1) for agentID in self._agentsIndices}
        self._currentAgentStatePar        = self._allAgentsStakedStateParDict[self._agentID]
        
        self._allAgentsStakedStatePar     = ca.vertcat(*[self._allAgentsStakedStateParDict[agentID] for agentID in self._agentsIndices]) # stacked version of the dictionary
        self._tPar                        = self._opti.parameter(1)
        
        
    def addFormulas(self,formulas : list[STLformula]|STLformula) -> None :
        """ Set the formulas for the edge that has to be respected by the edge. Input is expected to be a list  """
        
        # this is the case you only have one formula
        if not isinstance(formulas,list) :
            if isinstance(formulas,STLformula) :
                # set the source node pairs of this node
                if (formulas.sourceNode != self._agentID) and (formulas.targetNode != self._agentID) :
                    raise NotImplementedError(f"Seems that one or more of the inserted formulas do not involve the state of the current agent. Indeed agent index is {self._agentID}, but formula is defined over edge ({formulas.edgeTuple})")
                else :
                    self._STLformulas.append(formulas) # adding a single formula
                    
            else :
                raise Exception("please enter a valid STL formula object or a a list of STLformula objects")
                
        else :
            for formula in formulas :
                if not isinstance(formula,STLformula) :
                    raise Exception("Some of the given formulas are not STLformula objects. please revise your input")
                else :
                    if (formula.sourceNode != self._agentID) and (formula.targetNode != self._agentID) :
                        raise NotImplementedError(f"Seems that one or more of the inserted formulas do not involve the state of the current agent. Indeed agent index is {self._agentID}, but formula is defined over edge ({formula.edgeTuple})")
                    else :
                         self._STLformulas.append(formula) # adding a single formula
        
   
    def _setSingleLogExpConstraint(self,initialSystemState:dict[(int,np.ndarray)]) :
        """ here you need to give the iniital state of the whole multi agent system"""
        
        dummy = ca.MX.sym("x",1)
        alpha = ca.Function("alpha",[dummy],[dummy]) # simple linear
        
        if self._previouslyInitalised :
            raise NotImplementedError("this controller was previously initialised. Please create a new one if you want to change the constraints inside the controller. Chaging constraints dynamically is not supported yet")

        
        self._control  = self._opti.variable(self._stateSpaceDim,1) # taskes the dimension of the control input
        # create the state variable
        if len(self._STLformulas) == 0:
            self._barrierConstraints = []
            return 
        
  
        sum = 0
        eta = 1
        for formula in self._STLformulas :
           
           if (not formula.isParametric) and (not formula.predicate._isApproximated): #case in which the non parameteric formula was left pristine
                predicateFunction = formula.predicate.function 
                # change the sign to have safe set in h(x)>0 (it is all convex so no problem changing the sign)
                predicateFunction = ca.Function("predicateFunction",[self._agentStateVar],[-predicateFunction(self._agentStateVar)])
                # compute value of the function for the initial conditions 
                barrierFunction = self._computeSymbolicBarrier(predicateFunction = predicateFunction,initialAgentState = initialAgentState,initialNeigboursState = initialNeigboursState,formula = formula)
                sum += -eta*ca.exp(barrierFunction)         
           
           else :
                # if the formula is parameteric we take each plane and we set  it as a barrier constraint. Remember that the solution of the parameteric problem must have been found 
                A,b = formula.computeLinearHypercubeRepresentation(sourceNode=formula.sourceNode,targetNode=formula.targetNode) # this are two parameters now
                try :
                    bSol     = globalOptimizer.value(b)
                    Asol     = globalOptimizer.value(A)
                    rows,cols = np.shape(Asol)
                except :
                    raise NotImplementedError("seems like you inserted a parameteric formula, but a solution for it was not found. Try solving a task decomposition and then reapplying the same formula")
                
                linearInequalities = A@self._agentStateVar-bSol
                # remember that before this passage the safe set is defied as Ax-b <=0. Now we want to shift this to Ax-b>=0 and for this reason we are chaging the sign. Also note how do we add a barrier for each face of the cuboid
                for jj in range(rows):
                    predicateFunction = ca.Function("predicateFunction",[self._agentStateVar],[-linearInequalities[jj]]) 
                    barrierFunction = self._computeSymbolicBarrier(predicateFunction = predicateFunction,initialAgentState = initialAgentState,initialNeigboursState = initialNeigboursState,formula = formula)
                    sum += -eta*ca.exp(barrierFunction)    
        
        
        logSumExp = -1/eta*ca.log(sum)
        finalBarrierToBeScaled = ca.Function("tobescaled",[self._allAgentsStakedStateVar,self._timeVar,self._scaleFactorVar],[logSumExp])
        
        # now we find the value of scale factor that renders the barrier positive
        initialSystemStackedState = np.hstack([initialSystemState[agentID] for agentID in self._agentsIndices]) # possibly to be changed 
        
        factorRange = np.arange(1,20,0.2) # all the agents will arrive at the same value for this parameter bacuse they all share the shole system state at the initial time
        t0 = 0
        finalScaleFactor = None
        for factor in factorRange:
            value = finalBarrierToBeScaled(initialSystemStackedState,t0,factor)
            if value >=0 :
                finalScaleFactor = value 
        if finalScaleFactor == None :
            raise Exception("it was not possible to find a valid scale factor for the proposed barrier please increase the range of search inside the class definition")
            
    
        finalBarrier = finalBarrierToBeScaled(self._allAgentsStakedStateVar,self._timeVar,finalScaleFactor)
        
        nablaXi     = ca.jacobian(finalBarrier,self._agentStateVar)
        nablaX      = ca.jacobian(finalBarrier,self._allAgentsStakedStateVar)
        loadSharing = ca.norm_2(nablaXi)**2/ca.norm_2(nablaX)**2
        
        dbdt        = ca.jacobian(finalBarrier,self._timeVar)
        
        barrierConstraint = nablaXi.T@self._control +  (dbdt + alpha(finalBarrier))*loadSharing
        
        # now make it a function of the parameters 
        barrierConstraintFun     = ca.Function("barrierConstraintFun",[self._allAgentsStakedStateVar,self._timeVar],[barrierConstraint])
        self._barrierConstraints += [barrierConstraintFun(self._allAgentsStakedStatePar,self._control,self._tPar)>=0]
        
        
        self._previouslyInitalised = True    
        
   
    def _computeSymbolicBarrier(self,predicateFunction : ca.Function,initialAgentState:np.ndarray,initialNeigboursState:dict[int,np.ndarray],formula:STLformula) :
        """ return a barrier that is a variable of all the agents state, the scale factor and the time """
        target = formula.targetNode
        source = formula.sourceNode 
        
        #TODO: time to satisfaction always assumed to be initial time for now. We need to change this
        timeSatisfaction = formula.timeInterval.a
        # TODO: we will have to check the time of satisfaction with the other formulas
        # this is future work. For now we only consider that we satisfy at the first time instant
            
        if formula.temporalOperator == "always" :
            timeToRemotion = formula.timeInterval.b # you need to conclude over the full time interval
        else :
            timeToRemotion = timeSatisfaction # the timeSatisfaction is also the time in which you can remove the constraint ofr an eventually 
        
        # REMEMBER : for a single agent specification the predicate is defined over the agent stae. On the other hand for an edge specifcation the predicate is defiuned over the edge. State and Edge have the same dimensions.
        if target == source :
            h0           = predicateFunction(initialAgentState) # initial value of the barrier. At the beginning it is commonly negative because you do not satisfy the barrier at the beginning. So this is the reason of the minus sign
        
        else :
            if self._agentID == target :
            
                initialEdgeState = initialAgentState - initialNeigboursState[source]
                h0 = predicateFunction(initialEdgeState) # initial value of the barrier. At the beginning it is commonly negative because you do not satisfy the barrier at the beginning. So this is the reason of the minus sign
        
            else : # reveresed edge
                
                initialEdgeState = initialNeigboursState[target] - initialAgentState
                h0 = predicateFunction(initialEdgeState) # initial value of the barrier. At the beginning it is commonly negative because you do not satisfy the barrier at the beginning. So this is the reason of the minus sign
        
        # we create now the time transient funciton
        # this time transient will lift our predicate so that we are positive at the beginning
        if h0 >= 0 : # you are already inside the barrier so no need to create a gamma function 
            gamma = ca.Function("gamma",[self._timeVar,self._scaleFactorVar],[0])
        else :   
            h0       = -self._scaleFactorVar*h0 # now we make h0 positive
            slope    = -h0/timeSatisfaction # negative slope
            gamma    = ca.Function("gamma",[self._timeVar,self._scaleFactorVar],[ca.if_else(self._timeVar<=timeSatisfaction,h0+self._timeVar*slope,0)]) # piece wise linear function
        
        activationFunction = ca.if_else(self._timeVar<=timeToRemotion,1.,0.)
        
        if source == target :  # single agent specification
            barrier  = activationFunction(self._timeVar)*(predicateFunction(self._agentStateVar) + gamma(self._timeVar))
        else : # case of the edge 

            if target == self._agentID :
                eijVar  = self._agentStateVar - self._allAgentsStakedStateVar[source]
                barrier  = activationFunction(self._timeVar)*(predicateFunction(eijVar) + gamma(self._timeVar))
                
            else :
                eijVar  = self._allAgentsStakedStateVar[target] - self._agentStateVar 
                barrier  = activationFunction(self._timeVar)*(predicateFunction(eijVar) + gamma(self._timeVar))
        
        return barrier
    
    def computeControlInput(self,currentNeigboursState:dict[(int,np.ndarray)],currentAgentState:np.ndarray,time:float) :
        """only the state of the neigbours is really required. The other states can be set to zero since they don't impact """
        if not len(self._STLformulas)== 0 :
            for key,neigbourParametericState in self._neigboursStatePar.items() :
                try :
                    self._opti.set_value(neigbourParametericState,currentNeigboursState[key]) # assign each paramatric state to the 
                except KeyError :
                    raise ValueError("One or many of the neigbours give are not in the neigbours list. Please verify that the state of the neigbours given are correct")
            
        
            self._opti.set_value(self._currentAgentStatePar,currentAgentState)
            self._opti.set_value(self._tPar,time)
        
        
        try :
            sol = self._opti.solve()
            return sol.value(self._control)
        except :
            raise RuntimeError("It was not possible to find a solution")
            
            
    def setUp(self,initialNeigboursState:dict[(int,np.ndarray)],initialAgentState= np.ndarray,allowSlackSatisfaction = False) :
        
        
        self._setConstraints(initialNeigboursState = initialNeigboursState , initialAgentState = initialAgentState)
        
        if len(self._barrierConstraints) != 0 :
            cost = self._control.T@self._control + 0*self._previousControlInputPar.T@self._control 
        else :
            cost = self._control.T@self._control 
            
        if allowSlackSatisfaction :
            if not len(self._STLformulas)==0 :
                epsilon = self._opti.variable(1)
                for constraint in self._barrierConstraints :
                    self._opti.subject_to(constraint >=-epsilon) # add all the constraints
                    self._opti.subject_to(epsilon>=0)
                cost += 10000*epsilon**2
        else :
            if not len(self._STLformulas)==0 :
                for constraint in self._barrierConstraints :
                    self._opti.subject_to(constraint >=0) # add all the constraints
  
        self._opti.minimize(cost) 
        # self._opti.solver("ipopt",{"calc_lam_p":False,"print_time":0},{"print_level":0})
        self._opti.solver('sqpmethod',{'qpsol':'qrqp','print_out':False,'print_time':False,"print_header":False,"verbose":False,'error_on_fail':False,'qpsol_options':{'print_out':False,"print_iter":False,"print_time":False,'verbose':False,'print_info':False}})
        #self._opti.solver("qpoases")


