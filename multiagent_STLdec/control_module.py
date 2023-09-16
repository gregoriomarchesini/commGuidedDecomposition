from multiagent_STLdec.decomposition_module import *
from multiagent_STLdec.predicate_builder_module import *
import casadi as ca


# TODO: add interface for more complex dynamics compared to a single 2D integrator
class STLController():
    """STL controller class"""
    def __init__(self,agentIndex : int,neigbours:list[int]) -> None:
        
        self._STLformulas        : list[STLformula] =  []
        self._agentDynamics      : ca.Function      = None 
        self._agentIndex         : int              = agentIndex
        self._opti               : ca.Opti          = ca.Opti()
        self._neigbours          : list[int]        = neigbours
        
        self._barrierConstraints :list        = []         # barrier constraint
        self._barrierGradients   :list        =[]
        
        
        self._previousControlInput            = None
        self._previousState                   = None
        
        self._previouslyInitalised = False
        
        
        
    def addFormulas(self,formulas : list[STLformula]|STLformula) -> None :
        """ Set the formulas for the edge that has to be respected by the edge. Input is expected to be a list  """
        
        # this is the case you only have one formula
        if not isinstance(formulas,list) :
            if isinstance(formulas,STLformula) :
                # set the source node pairs of this node
                if (formulas.sourceNode != self._agentIndex) and (formulas.targetNode != self._agentIndex) :
                    raise NotImplementedError(f"Seems that one or more of the inserted formulas do not involve the state of the current agent. Indeed agent index is {self._agentIndex}, but formula is defined over edge ({formulas.edgeTuple})")
                else :
                    self._STLformulas.append(formulas) # adding a single formula
                    
            else :
                raise Exception("please enter a valid STL formula object or a a list of STLformula objects")
                
        else :
            for formula in formulas :
                if not isinstance(formula,STLformula) :
                    raise Exception("Some of the given formulas are not STLformula objects. please revise your input")
                else :
                    if (formula.sourceNode != self._agentIndex) and (formula.targetNode != self._agentIndex) :
                        raise NotImplementedError(f"Seems that one or more of the inserted formulas do not involve the state of the current agent. Indeed agent index is {self._agentIndex}, but formula is defined over edge ({formula.edgeTuple})")
                    else :
                         self._STLformulas.append(formula) # adding a single formula
        
    def _setConstraints(self,initialNeigboursState:dict[(int,np.ndarray)],initialAgentState= np.ndarray):
        """ Fpr now the barriers are computed assuming a simple integrator system. We will havw to change this in the future"""
        
        if self._previouslyInitalised :
            raise NotImplementedError("this controller was previously initialised. Please create a new one if you want to change the constraints inside the controller. Chaging constraints dynamically is not supported yet")

        for key in initialNeigboursState :
            if not key in self._neigbours :
                raise ValueError("at least one of the neigbours given is not in the neigbours list. Please update the neigbours list")
        
        stateSpaceDim = len(initialAgentState)
        self._control  = self._opti.variable(stateSpaceDim,1) # taskes the dimension of the control input
        # create the state variable
        if len(self._STLformulas) == 0:
            self._barrierConstraints = []
            return 
            
        
        # set up all the parameters 
        stateSpaceDim                 = self._STLformulas[0].stateSpaceDimension
        self._currentAgentStatePar    = self._opti.parameter(stateSpaceDim,1)
        self._neigboursStatePar       = {neigbour:self._opti.parameter(stateSpaceDim,1) for neigbour in self._neigbours}
        self._tPar                    = self._opti.parameter(1)

       
        # self._previousControlInputPar = self._opti.parameter(stateSpaceDim,1)
        # self._previousStatePar        = self._opti.parameter(stateSpaceDim,1)
        # self._previousState           = initialAgentState
        #TODO:eliminate this
        
        stateVar   = ca.MX.sym("stateVar",stateSpaceDim,1)
        controlVar = ca.MX.sym("stateVar",stateSpaceDim,1)
        
        self._previousControlInput    = np.zeros((stateSpaceDim,1))
        
        agentStateVar    = ca.MX.sym("xi",stateSpaceDim,1) # state variable for the agents
        
        for formula in self._STLformulas :
           
           if not formula.isParametric :
                predicateFunction = formula.predicate.function 
                # change the sign to have safe set in h(x)>0 (it is all convex so no problem changing the sign)
                predicateFunction = ca.Function("predicateFunction",[agentStateVar],[-predicateFunction(agentStateVar)])
                # compute value of the function for the initial conditions 
                self._addBarrierConstraint(predicateFunction = predicateFunction,initialAgentState = initialAgentState,initialNeigboursState = initialNeigboursState,formula = formula)
           else :
                # if the formula is parameteric we take each plane and we set  it as a barrier constraint. Remember that the solution of the parameteric problem must have been found 
                A,b = formula.computeLinearHypercubeRepresentation(sourceNode=formula.sourceNode,targetNode=formula.targetNode) # this are two parameters now
                try :
                    bSol     = globalOptimizer.value(b)
                    Asol     = globalOptimizer.value(A)
                    rows,cols = np.shape(Asol)
                except :
                    raise NotImplementedError("seems like you inserted a parameteric formula, but a solution for it was not found. Try solving a task decomposition and then reapplying the same formula")
                
                linearInequalities = A@agentStateVar-bSol
                # remember that before this passage the safe set is defied as Ax-b <=0. Now we want to shift this to Ax-b>=0 and for this reason we are chaging the sign. Also note how do we add a barrier for each face of the cuboid
                for jj in range(rows):
                    predicateFunction = ca.Function("predicateFunction",[agentStateVar],[-linearInequalities[jj]]) 
                    self._addBarrierConstraint(predicateFunction = predicateFunction,initialAgentState = initialAgentState,initialNeigboursState = initialNeigboursState,formula = formula)
        
        self._previouslyInitalised = True       
    
    def _addBarrierConstraint(self,predicateFunction : ca.Function,initialAgentState:np.ndarray,initialNeigboursState:dict[int,np.ndarray],formula:STLformula) :
        """ Predicate function must be created such that the the predicate is positive when the agent is inside its superlevel set. Remember that all the predicates coming from the convexPredicate object are wih the opposit sign"""
        
        
        
        stateSpaceDim    = formula.stateSpaceDimension
        timeVar          = ca.MX.sym("dd",1)
        dummyControlVar  = ca.MX.sym("control",stateSpaceDim,1)
        agentStateVar    = ca.MX.sym("xi",stateSpaceDim) # state variable for the agents
        neigbourStateVar = ca.MX.sym("xj",stateSpaceDim) # state variable for the neighbour
            
            
        # create a linear alpha function. For now we use the barrier itself. but it theory alpha can be any K-class functions.
        alpha    = ca.Function("linear",[timeVar],[timeVar])
        # alpha    = ca.Function("quadratic",[dummyScalarVar],[dummyScalarVar**2])
        # alpha    = ca.Function("cubic",[dummyScalarVar],[dummyScalarVar**3])
        # alpha    = ca.Function("square",[dummyScalarVar],[dummyScalarVar*(1/2)])
        
        
        
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
            if self._agentIndex == target :
            
                initialEdgeState = initialAgentState - initialNeigboursState[source]
                h0 = predicateFunction(initialEdgeState) # initial value of the barrier. At the beginning it is commonly negative because you do not satisfy the barrier at the beginning. So this is the reason of the minus sign
        
            else : # reveresed edge
                
                initialEdgeState = initialNeigboursState[target] - initialAgentState
                h0 = predicateFunction(initialEdgeState) # initial value of the barrier. At the beginning it is commonly negative because you do not satisfy the barrier at the beginning. So this is the reason of the minus sign
        
        # we create now the time transient funciton
        # this time transient will lift our predicate so that we are positive at the beginning
        if h0 >= 0 : # you are already inside the barrier so no need to create a gamma function 
            gamma = ca.Function("gamma",[timeVar],[0])
            gammaDot = ca.Function("gammaDot",[timeVar],[0])
            
        else :   
            scaleFactor = 1.3 # >=1 to give some margin but better not 1 otherwise you barrier will start from avalue of zero
            h0          = -scaleFactor*h0 # now we make h0 positive
        
            # exponential decay
            # epsilon  = 10**-2 # small arbitrary value so that the exponential goes to zero at this time
            # beta     = -1/timeSatisfaction* ca.log(epsilon/h0)
            # gamma    = ca.Function("gamma",[tVar],[h0*ca.exp(-beta*tVar)-epsilon])   # exponential goes from hBar to 0 in a time equal to timeSatisfaction
            # gammaDot = ca.Function("gammaDot",[tVar],[-beta*h0*ca.exp(-beta*tVar)]) # derivative alreadu given explicitly
            slope    = -h0/timeSatisfaction # negative slope
            gamma    = ca.Function("gamma",[timeVar],[ca.if_else(timeVar<=timeSatisfaction,h0+timeVar*slope,0)]) # piece wise linear function
            gammaDot = ca.Function("gammaDot",[timeVar],[ca.if_else(timeVar<=timeSatisfaction,slope,0)]) # derivative of gamma already given explicitly for the barrier constraint
        
        barrier  = ca.Function("barrierFunction",[agentStateVar,timeVar],[predicateFunction(agentStateVar) + gamma(timeVar)])

        
        # create an activation funcition for this constraints
        if formula.temporalOperator == "always" :
            activationFunction = ca.Function("activation",[timeVar],[ca.if_else(timeVar<=timeToRemotion,1.,0.)])
        else :
            # if you change the time to satisfaction you will have also to
            activationFunction = ca.Function("activation",[timeVar],[ca.if_else(timeVar<=timeToRemotion,1.,0.)])
        
        if source == target :  # single agent specification
            
            nablaXi = ca.Function("jaccc",[agentStateVar],[ca.jacobian(predicateFunction(agentStateVar),agentStateVar)])
            self._barrierGradients += [ca.jacobian(predicateFunction(agentStateVar),agentStateVar)]
            # bdot(x,t) + nabla_xi b(x,t)^T * u + alpha(b(x,t))) >=-0
            barrierCondition  = ca.Function("barrierCondition",[agentStateVar,timeVar,dummyControlVar],[
                    activationFunction(timeVar)*( ca.dot(nablaXi(agentStateVar).T,dummyControlVar) + gammaDot(timeVar) + alpha(barrier(agentStateVar,timeVar)))])
        
            self._barrierConstraints += [barrierCondition(self._currentAgentStatePar,self._tPar,self._control)]
            self._barrierGradients +=[nablaXi(self._currentAgentStatePar)]
        else : # case of the edge 

            if target == self._agentIndex :
                eijVar  = agentStateVar - neigbourStateVar
                nablaXi = ca.Function("jaccc",[agentStateVar,neigbourStateVar],[ca.jacobian(predicateFunction(eijVar),agentStateVar)])
                
                nablaXNorm  = ca.Function("jaccc",[agentStateVar,neigbourStateVar],[ca.norm_2(ca.jacobian(predicateFunction(eijVar),ca.vertcat(agentStateVar,neigbourStateVar)))**2])  
                nablaXiNorm = ca.Function("jaccc",[agentStateVar,neigbourStateVar],[ca.norm_2(ca.jacobian(predicateFunction(eijVar),agentStateVar))**2])  
                
                loadSharingFunction = ca.Function("loadSharing",[agentStateVar,neigbourStateVar],[nablaXiNorm(agentStateVar,neigbourStateVar)/nablaXNorm(agentStateVar,neigbourStateVar)])
                
                # bdot(x,t) + (nabla_xi b(x,t)^T * u + alpha(b(x,t)))eta_sharing >=-0
                barrierCondition    = ca.Function("barrierCondition",[agentStateVar,neigbourStateVar,timeVar,dummyControlVar],[
                    activationFunction(timeVar)*( ca.dot(nablaXi(agentStateVar,neigbourStateVar).T,dummyControlVar) + ( gammaDot(timeVar) + alpha(barrier(eijVar,timeVar)))*loadSharingFunction(agentStateVar,neigbourStateVar))])
                
                # now take the gradient of the function 
                self._barrierConstraints += [barrierCondition(self._currentAgentStatePar,self._neigboursStatePar[source],self._tPar,self._control)]
                self._barrierGradients +=[nablaXi(self._currentAgentStatePar,self._neigboursStatePar[source])]
            
            
            else :
                eijVar  = neigbourStateVar - agentStateVar # note how here it is reversed
            
                nablaXi = ca.Function("jaccc",[agentStateVar,neigbourStateVar],[ca.jacobian(predicateFunction(eijVar),agentStateVar)])
                
                nablaXNorm  = ca.Function("jaccc",[agentStateVar,neigbourStateVar],[ca.norm_2(ca.jacobian(predicateFunction(eijVar),ca.vertcat(agentStateVar,neigbourStateVar)))**2])
                nablaXiNorm = ca.Function("jaccc",[agentStateVar,neigbourStateVar],[ca.norm_2(ca.jacobian(predicateFunction(eijVar),agentStateVar))**2])  
                
                
                loadSharingFunction = ca.Function("loadSharing",[agentStateVar,neigbourStateVar],[nablaXiNorm(agentStateVar,neigbourStateVar)/nablaXNorm(agentStateVar,neigbourStateVar)])
                
                # bdot(x,t) + (nabla_xi b(x,t)^T * u + alpha(b(x,t)))eta_sharing >=-0
                barrierCondition    = ca.Function("barrierCondition",[agentStateVar,neigbourStateVar,timeVar,dummyControlVar],[
                    activationFunction(timeVar)*( ca.dot(nablaXi(agentStateVar,neigbourStateVar).T,dummyControlVar) + ( gammaDot(timeVar) + alpha(barrier(eijVar,timeVar)))*loadSharingFunction(agentStateVar,neigbourStateVar))])
        
                self._barrierConstraints += [barrierCondition(self._currentAgentStatePar,self._neigboursStatePar[target],self._tPar,self._control)]
                self._barrierGradients +=[nablaXi(self._currentAgentStatePar,self._neigboursStatePar[target])]
    
    
       
    def computeControlInput(self,currentNeigboursState:dict[(int,np.ndarray)],currentAgentState:np.ndarray,time:float) :
        
        if not len(self._STLformulas)== 0 :
            for key,neigbourParametericState in self._neigboursStatePar.items() :
                try :
                    self._opti.set_value(neigbourParametericState,currentNeigboursState[key]) # assign each paramatric state to the 
                except KeyError :
                    raise ValueError("One or many of the neigbours give are not in the neigbours list. Please verify that the state of the neigbours given are correct")
            
        
            self._opti.set_value(self._currentAgentStatePar,currentAgentState)
            self._opti.set_value(self._tPar,time)
       
        # self._opti.set_value(self._previousControlInputPar,self._previousControlInput)
        # self._opti.set_value(self._previousStatePar,self._previousState)
        
        try :
            sol = self._opti.solve()
            self._previousControlInput = sol.value(self._control) # save the current input as the previous control input 
            self._previousState = currentAgentState
            
            return sol.value(self._control)
        
        except :
            raise RuntimeError("It was not possible to find a solution")
            
        
        
        
        
    def setUp(self,initialNeigboursState:dict[(int,np.ndarray)],initialAgentState= np.ndarray,allowSlackSatisfaction = False) :
        
        
        self._setConstraints(initialNeigboursState = initialNeigboursState , initialAgentState = initialAgentState)
        
        cost = self._control.T@self._control 
        
        if allowSlackSatisfaction :
            if not len(self._STLformulas)==0 :
                for constraint in self._barrierConstraints :
                    epsilon = self._opti.variable(1)
                    self._opti.subject_to(constraint >=-epsilon) # add all the constraints
                    self._opti.subject_to(epsilon>=0)
                    cost += 100000*epsilon**2
        else :
            if not len(self._STLformulas)==0 :
                for constraint in self._barrierConstraints :
                    self._opti.subject_to(constraint >=0) # add all the constraints
  
        self._opti.minimize(cost) 
        #self._opti.solver("ipopt",{"calc_lam_p":False,"print_time":0},{"print_level":0})
        self._opti.solver('sqpmethod',{'qpsol':'qrqp','print_out':False,'print_time':False,"print_header":False,"verbose":False,'error_on_fail':False,'qpsol_options':{'print_out':False,"print_iter":False,"print_time":False,'verbose':False,'print_info':False}})


class Agent() :
    def __init__(self,initialState : np.ndarray,agentIndex : int,neigbours) -> None:
        
        self._initialState :np.ndarray = initialState
        self._agentIndex   : int = agentIndex
        self._neigbours    :list = neigbours
        self._controller   = STLController(agentIndex=self._agentIndex,neigbours=self._neigbours)
        
        self._currentState          = initialState
        self._currentNeigboursState = dict() 
        self._deltaT                = 0.01
        
        stateVar   = ca.MX.sym("stateVar",len(initialState))
        controlVar = ca.MX.sym("stateVar",len(initialState))
        
        self._dynamics  = ca.Function("singleIntergator",[stateVar,controlVar],[stateVar + self._deltaT*controlVar]) #TODO: we will have to do something more fancy. Note that also the controller will have to get this dynamics diuring the set up
        
        self._isInitialised = False 
        self._neigboursStatesUpdated = True
        
    @property
    def currentState(self):
        return self.currentState
    @property
    def currentControlInput(self) :
        return self._currentControlInput
    @property
    def timeStep(self):
        return self._deltaT
    
    @property
    def neigbours(self):
        return self._neigbours 
    
    def initializeController(self,initialNeigboursState : dict,formulas : list[STLformula],allowSlackSatisfaction = False) :
        
        self._currentNeigboursState = initialNeigboursState
        self._controller.addFormulas(formulas=formulas)
        self._controller.setUp(initialNeigboursState = initialNeigboursState,initialAgentState =  self._initialState,allowSlackSatisfaction=allowSlackSatisfaction) 
        self._isInitialised = True
        self._hasTasks = len(self._controller._STLformulas)
        
    def updateNeighboursState(self,neigboursState) :
        self._currentNeigboursState = neigboursState
        self._neigboursStatesUpdated = True # now you have updated information
           
    def printCurrentConstraintValue(self): 
        print("--------------------------------------------------------------------------------------------------")
        print("List of constraints evaluation")
        print("--------------------------------------------------------------------------------------------------")
        for kk,constraint in enumerate(self._controller._barrierConstraints) :
            print(f"constraint {kk} value :{self._controller._opti.debug.value(constraint)}")
        
    
    def printBarrierGradients(self) :
        for kk,grad in enumerate(self._controller._barrierGradients) :
                print(f"Gradient {kk} value :{self._controller._opti.debug.value(grad)}")
            
        
        
    
    def step(self,time)  :
        
        if not self._isInitialised :
            raise NotImplementedError("controller for this agent was not initialised")
        if (not self._neigboursStatesUpdated) and self._hasTasks :
            raise NotImplementedError("you did not update the state of the neigbours")
        
        try :
          self._currentControlInput    = self._controller.computeControlInput(currentNeigboursState =  self._currentNeigboursState,currentAgentState= self._currentState,time=time)
        except :
            print("--------------------------------------------------------------------------------------------------")
            print(f"It was not possible to cmpute the control input for agent {self._agentIndex}")
            print("Here is a complete list of the barrier constraints evaluation for the current agent state and time")
            print(f"agent state : {self._currentState}")
            print(f"current time : {time}")
            print(f"best solution found for control input : {self._controller._opti.debug.value(self._controller._control)}")
            print("List of constraints evaluation")
            print("--------------------------------------------------------------------------------------------------")
            for kk,constraint in enumerate(self._controller._barrierConstraints) :
                print(f"constraint {kk} value :{self._controller._opti.debug.value(constraint)}")
            print("List of barrer gradients")
            print("--------------------------------------------------------------------------------------------------")
            self.printBarrierGradients()
            
            
            
            raise RuntimeError("Solution not found")
        
        
        
        self._currentState           = self._dynamics(self._currentState,self._currentControlInput)
        self._neigboursStatesUpdated = False # so at the next step it is required that you update the state of the neigbours
        
        return self._currentState
       
        
        
        
        


