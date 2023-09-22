from multiagent_STLdec.decomposition_module import *
from multiagent_STLdec.predicate_builder_module import *
import casadi as ca


# TODO: add interface for more complex dynamics compared to a single 2D integrator
class STLController():
    """STL controller class"""
    def __init__(self,agentIndex : int,neigbours:list[int],stateSpaceDim:int,alphaMarginOnBarrier = 0) -> None:
        
        self._STLformulas        : list[STLformula] =  []
        self._agentDynamics      : ca.Function      = None 
        self._agentIndex         : int              = agentIndex
        self._opti               : ca.Opti          = ca.Opti("conic")
        self._neigbours          : list[int]        = neigbours
        
        self._stateSpaceDim      : int        = stateSpaceDim
        
        self._barrierConstraints  =  []         # barrier constraint
        self._barrierGradients    =  []
        self._barrierValues       =  []
        self._activationFunctions =  []
        
        
        self._initializationTime = 0
        
        if alphaMarginOnBarrier>= 0 :
            self._alphaMarginOnBarrier = alphaMarginOnBarrier #  non-negative 
        else :
            raise ValueError("Alpha marginb should be non negative")
        
        
        self._previouslyInitalised = False
        
        
        # variables for function creation
        self._timeVar                    = ca.MX.sym("dd",1)
        self._agentStateVar              = ca.MX.sym("xi",self._stateSpaceDim ) # state variable for the agents
        self._neigbourStateVar           = ca.MX.sym("xi",self._stateSpaceDim ) # state variable for the agents
        self._dummyScalar                = ca.MX.sym("scale",1) # it will serve to put the minimum approximation at a postive value
        self._controlVar                 = ca.MX.sym("xi",self._stateSpaceDim ) # state variable for the agents
        
        
        self._control  = self._opti.variable(stateSpaceDim,1) # taskes the dimension of the control input
        self._epsilon  = self._opti.variable(1)
        
        # set up all the parameters for optimization
        self._currentAgentStatePar       = self._opti.parameter(self._stateSpaceDim ,1)
        self._neigboursStatePar       = {neigbour:self._opti.parameter(stateSpaceDim,1) for neigbour in self._neigbours}
        self._tPar                       = self._opti.parameter(1)
        self._counter = 0
        self._epsilon = self._opti.variable(1)
        
        self._currentControlInput = np.zeros(self._stateSpaceDim)
        
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
        
        for key in initialNeigboursState :
            if not key in self._neigbours :
                raise ValueError("at least one of the neigbours given is not in the neigbours list. Please update the neigbours list")
        
        stateSpaceDim = len(initialAgentState)
        # create the state variable
        if len(self._STLformulas) == 0:
            self._barrierConstraints = []
            return 
            

       
        # self._previousStatePar        = self._opti.parameter(stateSpaceDim,1)
        # self._previousState           = initialAgentState
        #TODO:eliminate this
        
        
        self._agentStateVar    = ca.MX.sym("xi",stateSpaceDim,1) # state variable for the agents
        
        
        for formula in self._STLformulas :
           
            if (not formula.isParametric) and (not formula.predicate._isApproximated): #case in which the non parameteric formula was left pristine
                predicateFunction = formula.predicate.function 
                # change the sign to have safe set in h(x)>0 (it is all convex so no problem changing the sign)
                predicateFunction = ca.Function("predicateFunction",[self._agentStateVar],[-predicateFunction(self._agentStateVar)/10])
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
                
                linearInequalities = -1*(A@self._agentStateVar-bSol)
                # remember that before this passage the safe set is defied as Ax-b <=0. Now we want to shift this to Ax-b>=0 and for this reason we are chaging the sign. Also note how do we add a barrier for each face of the cuboid
                sum = 0
                eta = 10
                for jj in range(rows):
                    sum += ca.exp(-eta*linearInequalities[jj]) 
                    predicateFunction = ca.Function("predicateFunction",[self._agentStateVar],[linearInequalities[jj]/10]) 
                    self._addBarrierConstraint(predicateFunction = predicateFunction,initialAgentState = initialAgentState,initialNeigboursState = initialNeigboursState,formula = formula)
        
                # predicateFunction = ca.Function("predicateFunction",[self._agentStateVar],[-1/eta*ca.log(sum)]) 
                # self._addBarrierConstraint(predicateFunction = predicateFunction,initialAgentState = initialAgentState,initialNeigboursState = initialNeigboursState,formula = formula)
        self._previouslyInitalised = True       
    
    
    def _addBarrierConstraint(self,predicateFunction : ca.Function,initialAgentState:np.ndarray,initialNeigboursState:dict[int,np.ndarray],formula:STLformula) :
        """ Predicate function must be created such that the the predicate is positive when the agent is inside its superlevel set. Remember that all the predicates coming from the convexPredicate object are wih the opposit sign"""
         
        self._counter +=1
        if np.mod(self._counter,2) == 0 :
            alpha    = ca.Function("quadratic",[self._dummyScalar],[ca.if_else(self._dummyScalar <=0,200*self._dummyScalar ,10*self._dummyScalar -self._alphaMarginOnBarrier)]) #sim1
        else :
            alpha    = ca.Function("quadratic",[self._dummyScalar ],[ca.if_else((self._dummyScalar)<=0,
                                                                            200*self._dummyScalar,
                                                                            self._dummyScalar*2-self._alphaMarginOnBarrier)]) #sim1
        
        target = formula.targetNode
        source = formula.sourceNode 
        #TODO: time to satisfaction always assumed to be initial time for now. We need to change this
        timeSatisfaction = formula.timeOfSatisfaction
        timeOfRemotion = formula.timeOfRemotion 
        
        # REMEMBER : for a single agent specification the predicate is defined over the agent stae. On the other hand for an edge specifcation the predicate is defiuned over the edge. State and Edge have the same dimensions.
        if target == source :
            h0  = predicateFunction(initialAgentState) # initial value of the barrier. At the beginning it is commonly negative because you do not satisfy the barrier at the beginning. So this is the reason of the minus sign
        
        else :
            if self._agentIndex == target :
            
                initialEdgeState = initialAgentState - initialNeigboursState[source]
                h0 = predicateFunction(initialEdgeState) # initial value of the barrier. At the beginning it is commonly negative because you do not satisfy the barrier at the beginning. So this is the reason of the minus sign
        
            else : # reveresed edge
                
                initialEdgeState = initialNeigboursState[target] - initialAgentState
                h0 = predicateFunction(initialEdgeState) # initial value of the barrier. At the beginning it is commonly negative because you do not satisfy the barrier at the beginning. So this is the reason of the minus sign
        
        # we create now the time transient funciton
        # this time transient will lift our predicate so that we are positive at the beginning
        scaleFactor =  1*(1+(timeSatisfaction-self._initializationTime)/10)   #1.4 # >=1 to give some margin but better not 1 otherwise you barrier will start from avalue of zero
        
        if h0 <= 0 : # you are already inside the barrier so no need to create a gamma function 
            if timeSatisfaction==0 :
                raise ValueError("There is at least one formula that cannot be satisfied as the initial state is outside the safe set and the formula must hold from time zero. You need to move the inital state of the agent to the safe set if you want satisfaction from the inital time")
            gamma0  = scaleFactor*(1-h0) # now we make h0 positive
        else :
            gamma0  = scaleFactor*(1+h0)
        
        if timeSatisfaction<=self._initializationTime :
            print(f"fomula with id:{id(formula)} cannot be satisfied as the time of satisfaction is passed the initalization time. Initalization time is {self._initializationTime} and given satisfaction time is {timeSatisfaction} ")
            print("the formulas has the following specifics :")
            print(f"time interval : {formula.timeInterval.a,formula.timeInterval.b}")
            print(f"temporal operator : {formula.temporalOperator}")
            raise RuntimeError("Unsatisfiable formula")
        # exponential decay
        # epsilon  = 10**-2 # small arbitrary value so that the exponential goes to zero at this time
        # beta     = -1/timeSatisfaction* ca.log(epsilon/h0)
        # gamma    = ca.Function("gamma",[tVar],[h0*ca.exp(-beta*tVar)-epsilon])   # exponential goes from hBar to 0 in a time equal to timeSatisfaction
        # gammaDot = ca.Function("gammaDot",[tVar],[-beta*h0*ca.exp(-beta*tVar)]) # derivative alreadu given explicitly
        
        slope    = -gamma0/(timeSatisfaction-self._initializationTime) # negative slope
        tau      = - ca.log(1E-6/gamma0)/(timeSatisfaction-self._initializationTime)
        
        # gamma    = ca.if_else(self._timeVar<=timeSatisfaction,gamma0*ca.exp(-self._timeVar*tau)-1E-6,0) # piece wise linear function
        # gammaDot = ca.if_else(self._timeVar<=timeSatisfaction,-gamma0*tau*ca.exp(-self._timeVar*tau),0).printme(0) # derivative of gamma already given explicitly for the barrier constraint
        
        gamma    = ca.if_else(self._timeVar<=timeSatisfaction,gamma0+(self._timeVar-self._initializationTime)*slope,0) # piece wise linear function
        gammaDot = ca.if_else(self._timeVar<=timeSatisfaction,slope,0) # derivative of gamma already given explicitly for the barrier constraint
    
        # create an activation funcition for this constraints
        
        activationFunction = ca.if_else(self._timeVar<=timeOfRemotion,1.,0.)
        
        
        barrier  = ca.Function("barrierFunction",[self._agentStateVar,self._timeVar],[activationFunction*(predicateFunction(self._agentStateVar) + gamma)])
        
        if source == target :  # single agent specification
            
            
            nablaXi = ca.jacobian(predicateFunction(self._agentStateVar),self._agentStateVar)
            loadSharing = ca.if_else(ca.norm_2(nablaXi)>=0.000001,1,0)

            
            # bdot(x,t) + nabla_xi b(x,t)^T * u + alpha(b(x,t))) >=-0
            barrierCondition  = ca.Function("barrierCondition",[self._agentStateVar,self._timeVar,self._controlVar],[
                    activationFunction*( ca.dot(nablaXi.T,self._controlVar) + (gammaDot + alpha(barrier(self._agentStateVar,self._timeVar)))*loadSharing)])
        
            self._barrierConstraints += [barrierCondition(self._currentAgentStatePar,self._tPar,self._control)]
            self._barrierValues      += [barrier(self._currentAgentStatePar,self._tPar)]
        
        
        else : # case of the edge 

            if target == self._agentIndex :
                eijVar  = self._agentStateVar - self._neigbourStateVar
                
                nablaXi     = ca.jacobian(predicateFunction(eijVar),self._agentStateVar)
                nablaXNorm  = ca.norm_2(ca.jacobian(predicateFunction(eijVar),ca.vertcat(self._agentStateVar,self._neigbourStateVar)))**2
                nablaXiNorm = ca.norm_2(ca.jacobian(predicateFunction(eijVar),self._agentStateVar))**2
                loadSharingFunction = ca.if_else(nablaXNorm>=0.0000001,nablaXiNorm/nablaXNorm,0)
                
                
                # bdot(x,t) + (nabla_xi b(x,t)^T * u + alpha(b(x,t)))eta_sharing >=-0
                barrierCondition    = ca.Function("barrierCondition",[self._agentStateVar,self._neigbourStateVar,self._timeVar,self._controlVar],[
                    activationFunction*( ca.dot(nablaXi.T,self._controlVar) + ( gammaDot + alpha(barrier(eijVar,self._timeVar)))*loadSharingFunction)])
                
                # now take the gradient of the function 
                self._barrierConstraints += [barrierCondition(self._currentAgentStatePar,self._neigboursStatePar[source],self._tPar,self._control)]
                self._barrierValues       += [barrier(self._currentAgentStatePar-self._neigboursStatePar[source],self._tPar)]
            
            else :
                eijVar  = self._neigbourStateVar - self._agentStateVar # note how here it is reversed
                
                nablaXi = ca.jacobian(predicateFunction(eijVar),self._agentStateVar)
                nablaXNorm  = ca.norm_2(ca.jacobian(predicateFunction(eijVar),ca.vertcat(self._agentStateVar,self._neigbourStateVar)))**2
                nablaXiNorm = ca.norm_2(ca.jacobian(predicateFunction(eijVar),self._agentStateVar))**2
                loadSharingFunction = ca.if_else(nablaXNorm>=0.0000001,nablaXiNorm/nablaXNorm,0)
                
                 
                # bdot(x,t) + (nabla_xi b(x,t)^T * u + alpha(b(x,t)))eta_sharing >=-0
                barrierCondition    = ca.Function("barrierCondition",[self._agentStateVar,self._neigbourStateVar,self._timeVar,self._controlVar],[
                    activationFunction*( ca.dot(nablaXi.T,self._controlVar) + ( gammaDot + alpha(barrier(eijVar,self._timeVar)))*loadSharingFunction)])
        
                self._barrierConstraints += [barrierCondition(self._currentAgentStatePar,self._neigboursStatePar[target],self._tPar,self._control)]
                self._barrierValues      += [barrier(self._neigboursStatePar[target]-self._currentAgentStatePar,self._tPar)]
                activationFunction
    
    
    def computeControlInput(self,currentNeigboursState:dict[(int,np.ndarray)],currentAgentState:np.ndarray,time:float) :
      
        if not len(self._STLformulas)== 0 :
            for key,neigbourParametericState in self._neigboursStatePar.items() :
                try :
                    self._opti.set_value(neigbourParametericState,currentNeigboursState[key]) # assign each paramatric state to the 
                except KeyError :
                    raise ValueError("One or many of the neigbours give are not in the neigbours list. Please verify that the state of the neigbours given are correct")
            
        
            self._opti.set_value(self._currentAgentStatePar,currentAgentState)
            self._opti.set_value(self._tPar,time)
        # self._opti.set_value(self._previousStatePar,self._previousState)
        
        try :
            sol = self._opti.solve()
            self._currentControlInput = sol.value(self._control)
            
        except :
            pass
            # control input does not get updated and you just use the past one
        
        return self._currentControlInput
        
           
            
        
    def setUp(self,initialNeigboursState:dict[(int,np.ndarray)],initialAgentState:np.ndarray,initializationTime:float,allowSlackSatisfaction = False) :
        
        self._initializationTime = initializationTime
        if self._previouslyInitalised :
            self._clean()
        self._setConstraints(initialNeigboursState = initialNeigboursState , initialAgentState = initialAgentState)
        
        cost = self._control.T@self._control 
            
        if allowSlackSatisfaction :
            if not len(self._STLformulas)==0 :
                self._opti.set_initial(self._epsilon,0)
                for constraint in self._barrierConstraints :
                    self._opti.subject_to(constraint >=-self._epsilon) # add all the constraints
                    self._opti.subject_to(self._epsilon>=0)
                cost += 100*(1+self._epsilon)**2
        else :
            if not len(self._STLformulas)==0 :
                for constraint in self._barrierConstraints :
                    self._opti.subject_to(constraint >=0) # add all the constraints
                cost += 0*self._epsilon
                
        self._opti.minimize(cost) 
        #self.self._epsilon = self._opti.variable(1)anOutdatedFormulas(self) :
        
        self._opti.solver("qpoases",{"printLevel":"none"})
    
    
    
    
    
    def _cleanOutdatedFormulas(self) :
        self._STLformulas = [formula for formula in self._STLformulas if not(formula.timeOfRemotion<self._initializationTime)]
    
    def _defineNewOptimizer(self) :
        
        # clean the optimizer
        del self._opti 
        self._opti =  ca.Opti("conic")
        
        # set up all the parameters for optimization
        self._currentAgentStatePar = self._opti.parameter(self._stateSpaceDim ,1)
        self._neigboursStatePar    = {neigbour:self._opti.parameter(self._stateSpaceDim,1) for neigbour in self._neigbours}
        self._tPar                 = self._opti.parameter(1)
        self._counter = 0
        self._epsilon = self._opti.variable(1)
        self._control = self._opti.variable(self._stateSpaceDim)
        self._currentControlInput = np.zeros(self._stateSpaceDim)
        
        self._barrierConstraints  =  []         # barrier constraint
        self._barrierGradients    =  []
        self._barrierValues       =  []
        self._activationFunctions =  []
        
        
    def _clean(self):
        self._cleanOutdatedFormulas()
        self._defineNewOptimizer()
        




















class Agent() :
    def __init__(self,initialState : np.ndarray,agentIndex : int,neigbours) -> None:
        
        self._initialState :np.ndarray = initialState
        self._agentIndex   : int = agentIndex
        self._neigbours    :list = neigbours
        
        self._currentState          = initialState
        self._currentNeigboursState = dict() 
        self._deltaT                = 0.002
        self._alphaMargin  =  10*self._deltaT # 5 for sim1
        
        self._controller   = STLController(agentIndex=self._agentIndex,neigbours=self._neigbours,stateSpaceDim=len(initialState),alphaMarginOnBarrier=self._alphaMargin)
        
        stateVar   = ca.MX.sym("stateVar",len(initialState))
        controlVar = ca.MX.sym("stateVar",len(initialState))
        
        self._dynamics  = ca.Function("singleIntergator",[stateVar,controlVar],[stateVar + self._deltaT*controlVar]) #TODO: we will have to do something more fancy. Note that also the controller will have to get this dynamics diuring the set up
        
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
    
    def initializeController(self,initialNeigboursState : dict,formulas : list[STLformula],initializationTime:float,allowSlackSatisfaction = False) :
        
        
        self._currentNeigboursState           = initialNeigboursState
        self._controller.addFormulas(formulas = formulas)
        self._controller.setUp(initialNeigboursState  = initialNeigboursState,
                               initialAgentState      =  self._initialState,
                               initializationTime     = initializationTime,
                               allowSlackSatisfaction = allowSlackSatisfaction) 
        
        self._hasTasks = len(self._controller._STLformulas)
        
    
    def cleanController(self,initialNeigboursState : dict,initializationTime:float,allowSlackSatisfaction = False) :
        """This need to be called when you have already ba controller setted up"""
        if not self._controller._previouslyInitalised :
            raise Exception("controller for the urrent agent was not initalized before. Please run a first initialization before cleaning the controller")
        
        
        self._intializationTime = initializationTime
        self._currentNeigboursState = initialNeigboursState
        self._controller.setUp(initialNeigboursState = initialNeigboursState,
                               initialAgentState =  self._initialState,
                               initializationTime     = initializationTime,
                               allowSlackSatisfaction=allowSlackSatisfaction) 
        self._hasTasks = len(self._controller._STLformulas)
        
    
    
    def updateNeighboursState(self,neigboursState) :
        self._currentNeigboursState = neigboursState
        self._neigboursStatesUpdated = True # now you have updated information
           
    def printCurrentConstraintValue(self): 
        print("--------------------------------------------------------------------------------------------------")
        print(f"List of constraints evaluation for agent {self._agentIndex}")
        print("--------------------------------------------------------------------------------------------------")
        if len(self._controller._barrierConstraints) != 0 :
            for kk,constraint in enumerate(self._controller._barrierConstraints) :
                print(f"constraint {kk} value :{self._controller._opti.debug.value(constraint)}")
            print(f"time :{self._controller._opti.debug.value(self._controller._tPar)}")
            print(f"state :{self._controller._opti.debug.value(self._controller._currentAgentStatePar)}")
        
        else :
            print(f"Agent {self._agentIndex} does not have any constraint")
    
    def printCurrentBarrierValue(self) :
        print("--------------------------------------------------------------------------------------------------")
        print(f"List of barriers value for agent {self._agentIndex}")
        print("--------------------------------------------------------------------------------------------------")
        if len(self._controller._barrierValues) != 0 :
            for kk,barrier in enumerate(self._controller._barrierValues) :
                print(f"constraint {kk} value :{self._controller._opti.debug.value(barrier)}")   
                
            print(f"time :{self._controller._opti.debug.value(self._controller._tPar)}")
            print(f"state :{self._controller._opti.debug.value(self._controller._currentAgentStatePar)}")
        
        else :
            print(f"Agent {self._agentIndex} does not have any constraint")
    
    

    
    def step(self,time)  :
        
        if not self._controller._previouslyInitalised :
            raise NotImplementedError("controller for this agent was not initialised")
        if (not self._neigboursStatesUpdated) and self._hasTasks :
            raise NotImplementedError("you did not update the state of the neigbours")
        
       
        self._currentControlInput    = self._controller.computeControlInput(currentNeigboursState =  self._currentNeigboursState,currentAgentState= self._currentState,time=time)
        
        self._currentState           = self._dynamics(self._currentState,self._currentControlInput)
        self._neigboursStatesUpdated = False # so at the next step it is required that you update the state of the neigbours
        
        return self._currentState
       
    def getListOfBarrierValuesAtCurrentTime(self)  :
        if not self._controller._previouslyInitalised :
            raise NotImplementedError("controller for this agent was not initialised")
        if len(self._controller._barrierValues) != 0 :
            values = []
            for barrier in self._controller._barrierValues :
                values += [self._controller._opti.debug.value(barrier)]
            return values
        else :
            return []        
        
    def getListOfBarrierConstraintValuesAtCurrentTime(self)  :
        if not self._controller._previouslyInitalised :
            raise NotImplementedError("controller for this agent was not initialised")
        if len(self._controller._barrierConstraints) != 0 :
            values = []
            for constraint in self._controller._barrierConstraints :
                values += [self._controller._opti.debug.value(constraint)]
            return values
        else :
            return []      


