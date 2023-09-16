import numpy as np
from   matplotlib.patches import Ellipse
import casadi as ca
import itertools
from typing import Self


globalOptimizer = ca.Opti() # the global optimizer for the problem


class convexPredicateFunction() :
    """class for convx predicate functions"""
    def __init__(self,
                 stateSpaceDimension : int,
                 function : ca.Function = None ,
                 functionFlipped :ca.Function = None,
                 centerGuess : np.ndarray  = None,
                 computeApproximation = False,
                 sourceNode : int = None,
                 targetNode : int = None,) -> None:
        
        
        allNone = [function == None,not isinstance(centerGuess,np.ndarray),functionFlipped == None]
        allGiven = [function != None,isinstance(centerGuess,np.ndarray),functionFlipped != None]
        
        if not (all(allNone) or all(allGiven)) :
            raise ValueError("function, functionFlipped and centerGuess must be all given or none of them should be set")
        
        else :
            # TODO: put check on casadi funciton
            self._function        = function
            self._functionFlipped = functionFlipped # this is the funciton that you would call if the source and target node where to reversed.
            
            # TODO: put check on the ninput array
            self._centerGuess = centerGuess # superlevel set center guess. It just needs to be inside the superlevel set for a fast and good estimate of polytopic approximation
        
        
        if not(isinstance(sourceNode,int) or sourceNode==None ) or not(isinstance(targetNode,int) or targetNode==None) or not(isinstance(stateSpaceDimension,int)) :
            raise ValueError("targetNodesourceNode and state space dimension must be integers")
        else :
            self._sourceNode = sourceNode
            self._targetNode = targetNode
            self._dim = stateSpaceDimension
            
        
        if self._function !=None :
            self._isParametric = 0
            self._centerVar   = None
            self._nuVar       = None
            self._etaVar      = None
            
            
        else :
            if self._function ==None :
                # create hypercube parameters since the formula is parameteric
                self._centerVar = globalOptimizer.variable(self._dim,1) # casadi.Opti.variable() edge Casadi Variable for optimization
                self._nuVar     = globalOptimizer.variable(self._dim,1) # casadi.Opti.variable() nu variable (cuboid dimensions) for optimization
                self._etaVar    = ca.vertcat(self._centerVar,self._nuVar) # casadi.Opti.variable() [centerVar,nuVar] (just concatenation)
                self._isParametric = 1
            
        self._isApproximationAvailable     = False
        self._optimalApproximationvertices = None
        self._optimalApproximationCenter   = None # center of the zero level set
        self._optimalApproximationNu       = None # nu vector of the cuboid containing the dimneioson of the cuboid
        
        if not self._isParametric and computeApproximation :
           self.computeBestCuboidApproximation()
           self._isApproximationAvailable = True
        
    @property
    def function(self) :
        return self._function
    @property
    def sourceNode(self) :
        return self._sourceNode
    @property
    def targetNode(self) :
        return self._targetNode
    @property
    def edgeTuple(self):
        return (self._sourceNode,self._targetNode)
    @property
    def stateSpaceDimension(self) :
        return self._dim
    @property
    def optimalApproximationCenter(self) :
        return self._optimalApproximationCenter
    @property
    def optimalApproximationNu(self) :
        return self._optimalApproximationNu
    @property
    def centerVar(self) :
        return self._centerVar
    @property
    def nuVar(self) :
        return self._nuVar
    @property
    def etaVar(self) :
        return self._etaVar
    @property
    def isParametric(self):
        return self._isParametric
    
    @property
    def hasUndefinedDirection(self):
        return ((self.sourceNode==None) or self.targetNode==None) 
    
    
    def computeBestCuboidApproximation(self) :
        """computes the best cuboid approximation of the predicate function.
           In the future this might be approximated by some more complex zonotopes if needed
        """
        opti = ca.Opti()
        
        centerVarDummy = opti.variable(self._dim,1)
        nuVarDummy  = opti.variable(self._dim,1)
        
     
        cartesianProductSets         = [[-1,1],]*self._dim
        hypercubeVertices            = np.array(list(itertools.product(*cartesianProductSets))).T # All vertices of hypercube centered at the origin (Each vertex is a column)
        parametericHypercubeVertices = centerVarDummy + hypercubeVertices*nuVarDummy/2                # Each column of the matrix represents one of the vertices
        
        # set dimensions to be positive
        opti.subject_to(nuVarDummy>=np.zeros((self._dim,1)))
        # for each vertex set inclusion in the original set
        numVertices = 2**self._dim
        for jj in range(numVertices) :
            vertex = parametericHypercubeVertices[:,jj]
            opti.subject_to(self._function(vertex)<=0) # each vertex must be contained in the zero level set of the cpnvex function
            
        # cost is the inverse of the volume
        volume = 1
        for kk in range(self._dim) :
            volume = volume * nuVarDummy[kk]
        
        cost = self._function(1/volume) # it must be convex
        
        # give good initial guess to avaoid initial high values
        
        opti.set_initial(centerVarDummy,self._centerGuess)
        opti.set_initial(nuVarDummy,np.ones(self._dim)*1.5) # just to not start with 0
        
        opti.minimize(cost)
        p_opts = dict(print_time=False, verbose=False)
        s_opts = dict(print_level=0)
        opti.solver("ipopt", p_opts, s_opts)

        try  :
            opti.solve()
        except :
            raise Exception("Error occurred while computing cuboid approximation of predicate superlevel set. Plase verify the the predicate function is convex and that the starting guess is inside reasonable to find the minim of the function")
        
        self._optimalApproximationVertices = opti.value(parametericHypercubeVertices)
        self._optimalApproximationCenter   = opti.value(centerVarDummy) # center of the zero level set
        self._optimalApproximationNu       = opti.value(nuVarDummy) # nu vector of the cuboid containing the dimneioson of the cuboid
        
        # NOTE : for the flipped predicate the center will be the negative of the center for the current function! So there is no need to compute the optimal hypercube underapproximation for the flipped predicate
        
    def replaceWithApproximatePredicate(self) :
        """
        This function is called in case the original predicate needs to be replaced with an under approximation of it
        """
        if not self._isApproximationAvailable : # if the approximation was not computed before at the creation of the instance then create it now
            self.computeBestCuboidApproximation(self)
            self._isApproximationAvailable = True
        
        self._function,self._functionFlipped = cuboidPredicate(center=self._optimalApproximationCenter ,dimensions=self._optimalApproximationNu)
        
    def flip(self):
        """
        flips the predicate
        """
        if self._isParametric :
            raise NotImplementedError("Parametric formulas cannot be flipped")
        # flip the source and target node
        dummy = self.sourceNode
        self._sourceNode = self._targetNode 
        self._targetNode = dummy
        
        # flip the predicate
        dummy = self._function
        self._function = self._functionFlipped
        self._functionFlipped = dummy
        
        
        # flip the best approximation center (the vector nu is unchanged after flipping)
        if self._isApproximationAvailable :
            self._optimalApproximationVertices = - self._optimalApproximationVertices
            self._optimalApproximationCenter   = - self._optimalApproximationCenter
        
     

    def hypercubeVertices(self,source:int,target:int) :
        
        if self.hasUndefinedDirection :
            raise NotImplementedError("predicate has undefined direction. Only if you define a target and source you can obtain the hypercube vertices")
        if self._isParametric :
            cartesianProductSets         = [[-1,1],]*self._dim
            hypercubeVertices            = np.array(list(itertools.product(*cartesianProductSets))).T # All vertices of hypercube centered at the origin (Each vertex is a column)
            
            if (self.edgeTuple != (source,target)) and (self.edgeTuple != (target,source)) : # this happens if the edge is not the same at all
                raise NotImplementedError("the given source\\target pair does not match the edge of the predicate")
            
            # if the direction of request matches the direction of the parametric predicate the use the normal center
            elif self.edgeTuple == (source,target) :
                # the requested direction is the orginal direction of the predicate
               parametericHypercubeVertices =    self._centerVar + hypercubeVertices*self._nuVar/2    # Each column of the matrix represents one of the vertices
            
            else : # use the opposite direction center
                parametericHypercubeVertices = -1*self._centerVar + hypercubeVertices*self._nuVar/2   # Each column of the matrix represents one of the vertices
            
        else :
            raise NotImplementedError("formula is not parametric so you cannot quary any vertex through this function. If you wanted to get the best cuboid underappriximation, please check the property optimalApproximationVertices")
        
        return parametericHypercubeVertices
        
    
    
    def linearRepresentationHypercube(self,source:int,target:int) :
        """returns linear representation of the parameteric function as Ax<=b"""
        
        if self.hasUndefinedDirection :
            raise NotImplementedError("predicate has undefined direction. Only if you define a target and source you can obtain the hypercube vertices")
        if self.isParametric :
            if (self.edgeTuple != (source,target)) and (self.edgeTuple != (target,source)) : # this happens if the edge is not the same at all
                raise NotImplementedError("the given source\\target pair dies not match the edge of the predicate")
            
            # A(x-c) - b <= 0
            elif self.edgeTuple == (source,target) :
                A  = np.vstack((np.eye(self.stateSpaceDimension),-np.eye(self.stateSpaceDimension)))  # (face normals x hypercube stateSpaceDimension)
                Ac = A@self._centerVar
                d  = ca.vertcat(self.nuVar/2,self.nuVar/2)
                b  = Ac+d
                return A,b
            else :
                A  = np.vstack((np.eye(self.stateSpaceDimension),-np.eye(self.stateSpaceDimension)))  # (face normals x hypercube stateSpaceDimension)
                Ac = A@(-self._centerVar) # revert the center
                d  = ca.vertcat(self.nuVar/2,self.nuVar/2)
                b  = Ac+d
                return A,b
                
        else : 
            raise NotImplementedError("formula is not parametrichis method is aonly available for parameteric formulas")
    
    def addSourceTarget(self,source:int,target:int) -> None:
        self._sourceNode = source
        self._targetNode = target 
        
        
class timeInterval() :
    """time interval class"""
    # empty set is represented by a double a=None b = None
    def __init__(self,a:float|int = None,b:float|int=None) -> None:
        
        
        
        if any([a==None,b==None]) and (not all(([a==None,b==None]))) :
            raise ValueError("only empty set is allowed to have None Values in the interval")
        elif  any([a==None,b==None]) and (all(([a==None,b==None]))) : # empty set
            self._a = a
            self._b = b
        else :    
            # all the checks 
            if (not isinstance(a,float)) and  (not isinstance(a,int)) :
                raise ValueError("the input a must be a float or int")
            elif a<0 :
                raise ValueError("extremes of time interval must be positive")
            
            # all the checks 
            if (not isinstance(b,float)) and  (not isinstance(b,int)) :
                raise ValueError("the input b must be a float or int")
            elif b<0 :
                raise ValueError("extremes of time interval must be non negative")
            
            if a>b :
                raise ValueError("Time interval must be a couple of non decreasing time instants")
         
        self._a = a
        self._b = b
        
    @property
    def a(self):
        return self._a
    @property
    def b(self):
        return self._b
    
    @property
    def measure(self) :
        if self.isEmpty() :
            return None # empty set has measure None
        return self._b - self._a
    
    @property
    def aslist(self) :
        return [self._a,self._b]
    
    def isEmpty(self)-> bool :
        if (self._a == None) and (self._b == None) :
            return True
        else :
            return False
        
    def isSingular(self)->bool:
        a,b = self._a,self._b
        if a==b :
            return True
        else :
            return False
    
    def __lt__(self,timeInt:Self) -> Self:
        """strict subset relations self included in timeInt ::: Self < timeInt """
        a1,b1 = timeInt.a,timeInt.b
        a2,b2 = self._a,self._b
        
        if self.isEmpty() and (not timeInt.isEmpty()) :
            return True
        elif (not self.isEmpty()) and timeInt.isEmpty() :
            return False
        elif  self.isEmpty() and timeInt.isEmpty() : # empty set included itself
            return True
        else :
            if (a1<a2) and (b2<b1): # condition for intersectin without inclusion of two intervals
                return True
            else :
                return False
    
    def __eq__(self,timeInt:Self) -> bool:
        """ equality check """
        a1,b1 = timeInt.a,timeInt.b
        a2,b2 = self._a,self._b
        
        if a1 == a2 and b1 == b2 :
            return True
        else :
            return False
        
    def __ne__(self,timeInt:Self) -> bool :
        """ inequality check """
        a1,b1 = timeInt.a,timeInt.b
        a2,b2 = self._a,self._b
        
        if a1 == a2 and b1 == b2 :
            return False
        else :
            return True
        
    def __le__(self,timeInt:Self) -> Self :
        """subset relations self included in timeInt  ::: Self < timeInt """
        
        a1,b1 = timeInt.a,timeInt.b
        a2,b2 = self._a,self._b
        
        if self.isEmpty() and (not timeInt.isEmpty()) :
            return True
        elif (not self.isEmpty()) and timeInt.isEmpty() :
            return False
        elif  self.isEmpty() and timeInt.isEmpty() : # empty set included itself
            return True
        else :
            if (a1<=a2) and (b2<=b1): # condition for intersectin without inclusion of two intervals
                return True
            else :
                return False
        
    def __truediv__(self,timeInt:Self) -> Self :
        """interval Intersection"""
        
        a1,b1 = timeInt.a,timeInt.b
        a2,b2 = self._a,self._b
        
        
        # the empty set is already in this cases since the empty set is included in any other set
        if timeInt<= self :
            return timeInterval(a =timeInt.a, b = timeInt.b)
        elif self<= timeInt :
            return timeInterval(a =self._a, b = self._b)
        else : # intersection case
            if (b1<a2) or (b2<a1) : # no intersection case
                return timeInterval(a = None, b = None)
            elif (a1<=a2) and (b1<=b2) :
                return timeInterval(a = a2, b = b1)
            else :
                return timeInterval(a = a1, b = b2)
                
        
    def epsilonRightShrink(self)-> None :
        """ machine precision shrinkig of the time interval on the right hand side"""
        
        if self.isEmpty() :
            return self
        else :
            dt = 2**(-52 + np.floor(np.log2(self._b))) # machine precision difference
            if self._b<= dt :
                raise Exception("Not possible to shrink. The right extreme would be negative after shrinking")
            else :
                self._b = self._b - dt
                
    
    def epsilonRightShrink(self)-> None :
        """ machine precision shrinkig of the time interval on the left hand side"""
        if self.isEmpty() :
            return self
        else :
            dt = 2**(-52 + np.floor(np.log2(self._b))) # machine precision difference
            self._b = self._b + dt
            
            
class STLformula( ) :
    
    def __init__(self,temporalOperator:str,timeinterval:timeInterval,predicate:convexPredicateFunction):
        
        """ 
        Input
        ------------------------------------------------------------------------------------------
        stateSpaceDimension         (int)          : 
        temporalOperator  (str)                    :
        timeinterval      (timeInterval)           :
        predicateFunction (convexPredicateFunction :
        
        Note : When a predicate function is not assigned, the attribute "isParametric" will be set to 1. This 
            entails that the predicate will be defined through optimization.
        """
        
        # if a predicate function is not assigned, it is considered that the predicate is parametric
        
        
        
        
        self._predicate              :convexPredicateFunction = predicate
        self._approximationAvailable :bool                    = False
        self._stateSpaceDimension    :int                     = self._predicate.stateSpaceDimension
       
            
        # set temporal prefix
        if temporalOperator!= "always" and temporalOperator!= "eventually" :
            raise Exception("Only 'eventually' and 'always' temporalPrefixs are accepted")
        else :
            self._temporalOperator   = temporalOperator # always or eventually
        
        if timeinterval.isEmpty() :
            raise NotImplementedError("Sorry, empty time intervals are not currently supported by this class")
        self._timeInterval  = timeinterval
            
    
    """STL formula class"""
    
    @property
    def predicateFunction(self):
        return  self._predicate.function
    @property
    def predicate(self):
        return self._predicate
    @property
    def temporalOperator(self):
        return self._temporalOperator
    @property
    def timeInterval(self):
        return self._timeInterval
    @property
    def centerVar(self):
        return self._predicate.centerVar        
    @property
    def nuVar(self):
        return self._predicate.nuVar          
    @property
    def etaVar(self):
        return self._predicate.etaVar         
    @property
    def stateSpaceDimension(self):
        return self._stateSpaceDimension      
    @property
    def isParametric(self):
        return self._predicate.isParametric  
    
    @property
    def edgeTuple(self) :
        return (self._predicate.sourceNode,self._predicate.targetNode)
    @property
    def sourceNode(self) :
        return self._predicate.sourceNode
    @property
    def targetNode(self):
        return self._predicate.targetNode
        
    
    def flip(self) :
        self._predicate.flip()
         
    def getHypercubeVertices(self,sourceNode,targetNode:int) -> list[ca.MX]:
        """ computes vertices of hypercube as function of the centerVar and the dimension vector nuVar"""
        
        return self._predicate.hypercubeVertices(source=sourceNode,target=targetNode)
        
    
    def computeLinearHypercubeRepresentation(self,sourceNode:int,targetNode:int) -> tuple[ca.MX,ca.MX] :
        """returns linear representation of the parameteric function as Ax<=b"""
        A,b = self._predicate.linearRepresentationHypercube(source=sourceNode,target=targetNode)
        return A,b

    def getConstraintFromInclusionOf(self,formula : Self) -> list[ca.MX]:
        """ returns inclusion constraints for "formula" inside the of the self formula instance """
        if not isinstance(formula,STLformula) :
            raise ValueError("input formula must be an instance of STL formula")
        # two cases :
        # 1) parameteric vs parametric
        # 2) parameteric vs non-parameteric
        numVertices = 2**self._stateSpaceDimension
        constraints = []
        
        if formula.isParametric and self.isParametric :
            
            source,target = self.edgeTuple
            # get the represenations of both formulas with the right verse of the edge
            vertices      = formula.getHypercubeVertices(sourceNode=source,targetNode=target)
            A,b           = self.computeLinearHypercubeRepresentation(sourceNode=source,targetNode=target)
            print("entering all parameteric case")
            constraints = []
            for jj in range(numVertices) : # number of vertices of hypercube is computable given the stateSpaceDimension of the problem
                constraints +=[ A@vertices[:,jj]-b<=np.zeros((self._stateSpaceDimension*2,1))]
                
        elif  formula.isParametric and (not self.isParametric) :
            
            source,target = self.edgeTuple # check the direction of definition
            
            if (formula.edgeTuple != self.edgeTuple) and (formula.edgeTuple != (target,source)) : # this happens if the edge is not the same at all
                raise NotImplementedError("It seems that you are trying to make an inclusion between two formulas that are not part of the same edge. This is not support for now")
            elif formula.edgeTuple != self.edgeTuple :
                self.flip() # change to flipped version to match the direction of the parametric formulas
                
            vertices   = formula.getHypercubeVertices(sourceNode=source,targetNode=target)
            constraints = []
            for jj in range(numVertices) : # number of vertices of hypercube is computable given the stateSpaceDimension of the problem
                constraints +=[ self._predicate.function(vertices[:,jj])<=0 ] 
            
        elif (not formula.isParametric) and self.isParametric :  
            raise NotImplementedError("Trying to include a non parameteric formula inside a parameteroc one. Not supported")
        
        return constraints
    


def ellipseThroughPoints(v1:np.ndarray,v2:np.ndarray,axesRatio:float =2) -> tuple[np.ndarray,np.ndarray,np.ndarray,Ellipse] :
    """ returns parameters of the ellipse with foci in v1 and v2 with the given ratin between the axis 
    Inputs 
    ------------------------------------------------------------------------------------------------
    
    v1        (np.array(2,)) : first vertex
    v2        (np.array(2,)) : second vecrtex
    axesRatio (float)        : ratio major to minor axis
    
    Output
    ------------------------------------------------------------------------------------------------
    P            (np.array(2,2))  : matrix representing representing the ellipse in matrix form (x-center)'P(x-center)<=1
    B            (np.array(2,2))  : matrix represting transformation from unit circle to ellipse  Bx + center with ||x||<=1
    center       (np.array(2,))   : center of the ellipse
    patchEllipse (Ellipse)        : patch of the ellipse that can be use to plot the ellipse in matpotlib
    """
    
    # flatten vectors
    v1 = v1.ravel()
    v2 = v2.ravel()
    
    if not(len(v1) ==2) or not(len(v2)==2) :
        raise Exception("Inputs must be 2D vectors but input vector have size " + str(len(v1) + " and " + str(len(v2)))) 
    
    if axesRatio <=1 :
        raise Exception("Axes ratio must be at least grater than 1")
    
    # compute center of the ellipse
    focalLineDirection = v1-v2        # direction of the focal line
    center             = v2+(v1-v2)/2 # center of the ellipse
    semiFocalDistance  = np.linalg.norm((v1-v2)/2) # focal distance
    # relation smifocalDistance (c), semimajor axis (a) and semiminor axis (b) c^2 = a^2-b^2
    semiMinorAxis = np.sqrt(semiFocalDistance**2/(axesRatio**2-1))  
    semiMajorAxis = semiMinorAxis*axesRatio
    # find the rotation angle of the focal line
    theta = np.arctan2(focalLineDirection[1],focalLineDirection[0])
    rotationMatrix = np.array([[np.cos(theta),-np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])
    # Ellipse Matrix
    P = rotationMatrix.T@np.diag([1/semiMajorAxis**2,1/semiMinorAxis**2])@rotationMatrix
    B = rotationMatrix@np.diag([semiMajorAxis,semiMinorAxis])
    patchEllipse = Ellipse(center,2*semiMajorAxis,2*semiMinorAxis,angle=np.rad2deg(theta))
    return P,B,center,patchEllipse


def computeEllipseMatrix(semiMajorAxis:float =1, semiMinorAxis:float =0.5, theta:float =0) -> tuple[np.ndarray,np.ndarray] :
    """ returns P and B matrix of an ellipse given as x'Px-1 and B matrix representing the ellipse in parameteric form as y=Bx with ||x||<=1
    
    Inputs 
    ------------------------------------------------------------------------------------------------
    
    semiMajorAxis (float): semimjor axis (x-direction)
    semiMinorAxis (float): semiminor axis (y-direction)
    theta (float)        : rotation angle of the ellipse in radians (positive counter-clock direction)
    
    
    Output
    ------------------------------------------------------------------------------------------------
    P (np.array(2,2))        : matrix representing representing the ellipse in matrix form (x-center)'P(x-center)<=1
    B (np.array(2,2))        : matrix represting transformation from unit circle to ellipse  Bx + center with ||x||<=1
    """
    
    if semiMajorAxis/semiMinorAxis <=1 :
        raise Exception("Axes ratio must be at least grater than 1")
    
    rotationMatrix = np.array([[np.cos(theta),-np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])
    # Ellipse Matrix
    P = rotationMatrix.T@np.diag([1/semiMajorAxis**2,1/semiMinorAxis**2])@rotationMatrix
    B = rotationMatrix@np.diag([semiMajorAxis,semiMinorAxis])
    return P,B

def maxDistancePredicate(stateSpaceDimension :int, maxDistance:float) -> tuple[convexPredicateFunction,convexPredicateFunction] :
    """returns a function consrainig maximum distance as ||x||-maxDistance**2<=0 twice. this is because the flipped versioon of this predicate is itself"""
    
    x = ca.MX.sym("x",stateSpaceDimension,1)
    function = ca.Function("predicate",[x],[(x).T@(x)-maxDistance**2])
   
    return  convexPredicateFunction(stateSpaceDimension=stateSpaceDimension,function=function,functionFlipped=function ,centerGuess=np.zeros(stateSpaceDimension)) 

def circlePredicate(stateSpaceDimension:int,radius:float,center:np.ndarray) -> convexPredicateFunction  :

    if center.ndim == 0 :
        center = center[:,np.newaxis]
    x = ca.MX.sym("x",stateSpaceDimension,1)
    function = ca.Function("predicate",[x],[(x-center).T@(x-center)-radius**2])
    functionFlipped = ca.Function("predicate",[x],[(x+center).T@(x+center)-radius**2])
    
    return  convexPredicateFunction(stateSpaceDimension=stateSpaceDimension,function=function,functionFlipped=functionFlipped ,centerGuess= center) # returns maximum diatance predicate
        
    
# some standard convex predicates
def ellipsoidPredicate(P: np.ndarray,center:np.ndarray) -> convexPredicateFunction:
    """returns a function warpping an ellipsoid predicate
    Inputs 
    ------------------------------------------------------------------------------------------------
    P      (np.array(n,n)) : symmetric matrix defyning elliposidal constrain like (x-center)'P(x-center)-1<0
    center (np.array(n,1)) : center of the ellipsoid predicate 

    Output 
    ------------------------------------------------------------------------------------------------
    predicate (casadi.Function) : predicate function evaluator --> predicate(x) = (x-center)'P(x-center)-1
    """
    n,m = np.shape(P)
    if n!=m :
        raise("matrix should be symmetric")
    if len(center)!=n :
        raise("variable 'center' has inappropriate dimension : matrix is " + str(n) +"x"+str(n)+", while vector is dimension" + str(len(center)))

    if center.ndim == 1 :
        center = center[:,np.newaxis] # put as column
    x = ca.SX.sym("x",n,1)
    function         = ca.Function("predicate",[x],[(x-center).T@P@(x-center)-1])   # retun the predicate 
    functionFlipped = ca.Function("predicate",[x],[(x-center).T@P@(x-center)-1])   # retun the predicate
    return convexPredicateFunction(stateSpaceDimension=len(center),function=function,functionFlipped= functionFlipped,centerGuess= center) 
        
           
def polytopicSetPredicate(center : np.ndarray, a_list : list[np.ndarray],distances:list[float],computeApprox:bool =True) -> convexPredicateFunction  :
    
    """ computes funciton h(x)<=0 for given polytope as a_i(x-c)<=dist_i 
    Inputs 
    ------------------------------------------------------------------------------------------------
    center (np.array(n,1))              : center of the polytope
    a_list ( list[np.ndarray])          : direction of the plane. Note that the inner of the plane will be taken in the opposite direction of this given direction. Also note that the vector will not be normalised
    distances     (list[float])         : distance of the plane from the given center
    
    Output 
    ------------------------------------------------------------------------------------------------
    h (casadi.Function) : predicate function evaluator. if the functon is negative for agiven point then you are inside the superlevel set.
    
    
    Try : define a plane with center in [0,0] and direction [1,0] and distance [0]. All the points [alpha,0] wth alpha>0
    will have a positive value of h([alpha,0]) meaning that you are outside the safe set. for alpha <0 you are inside instead
    """
    
    
    
    if center.ndim >1 :
        center = center.ravel()
    if len(a_list) == 0 :
        raise ValueError(" hyperplane direction-distance pairs is an empty list. Please provide some directions with relative distances from the center")
    
    if len(a_list) != len(distances) :
        raise ValueError("distance and planeDirections lists must have te same length")
    
    
    x = ca.MX.sym("x",*center.shape)
    listOfHyperplanes = []
    listOfFlippedHyperplanes = []
    for aVec,distance in zip(a_list,distances) :
        
        if len(aVec) != len(center) :
            raise ValueError("seems that at least one direction vector has different dimensions than the corresponding center vector. Please fix this")
        
        hyperplane        = (x-center).T@aVec[:,np.newaxis] - distance 
        flippedHyperplane = (x+center).T@aVec[:,np.newaxis] - distance 
        
        
        hyperplaneFun  = ca.Function("hyperplane",[x],[hyperplane]) # this defined an hyperplane function
        flippedHyperplaneFun = ca.Function("hyperplane",[x],[flippedHyperplane]) # this defined an hyperplane function
        
        listOfHyperplanes.append(hyperplaneFun)
        listOfFlippedHyperplanes.append(flippedHyperplaneFun)
    
    if len(listOfHyperplanes) != 1 :                                                                                                                                                                         
        # create a minimum approximation
        alpha = 3 # the bigger it is the better the approximation (but if it is too high you will have numericla problems)
        sum   = 0
        
        
        sumFlipped = 0
        # maximum approximation
        for hyperplaneFun,flippedHyperFun in zip(listOfHyperplanes,listOfFlippedHyperplanes) :
            sum = sum + ca.exp(alpha*hyperplaneFun(x))
            sumFlipped = sumFlipped + ca.exp(alpha*flippedHyperFun(x))
            
        h        = ca.Function("hyperplaneAprrox",[x],[1/alpha*ca.log(sum)])
        hFlipped = ca.Function("hyperplaneAprrox",[x],[1/alpha*ca.log(sumFlipped)])
        return convexPredicateFunction(stateSpaceDimension=len(center),function=h,centerGuess=center,functionFlipped=hFlipped,computeApproximation=computeApprox)
    else :
        h = listOfHyperplanes[0]
        hFlipped = listOfFlippedHyperplanes[0]
        return convexPredicateFunction(stateSpaceDimension=len(center),function=h,functionFlipped=hFlipped,centerGuess=center,computeApproximation=computeApprox)
        
def cuboidPredicate(center : np.ndarray,dimensions:list[float],computeApprox :bool = True) ->ca.Function  :
    # for now only 2d 
    
    if center.ndim >1 :
        center = center.ravel()
        
    for dim in dimensions :
        if dim < 0 : 
            raise ValueError("Some dimensions are not positive")
        
    if len(center) ==2 :
        if len(dimensions)!= 2 :
            raise ValueError("dimensions should have length 2 for a 2D cuboid")
        width,height = dimensions
        
        v1 = np.array([1,0])
        v2 = np.array([-1,0])
        v3 = np.array([0,1])
        v4 = np.array([0,-1])
        normals = [v1,v2,v3,v4]
        dists = [width/2,width/2,height/2,height/2]
        
        
        predicate:convexPredicateFunction = polytopicSetPredicate(center = center,a_list = normals, distances= dists,computeApprox=computeApprox)
        return predicate
        
        

if __name__ =="__main__" :
    pass
    # # simple one directional plane predicate function
    # h = polytopicSetPredicate(center = np.array([0,0]),planeDirections=[np.array([1.,0])],distances= [-2.0])   
    # print(h(np.array([-2,0])))
    
    
   
    # h = cuboidPredicate(center = np.array([0,0]),dimensions=[3,4])
    
    # print(h(np.array([0,-1.4])))
    # predicateObj= convexPredicateFunction(function=h,centerGuess=np.array([0,0]))
    # print(predicateObj._vertices)
    
    # a = ca.DM([1,2])
    # print(ca.norm_2(a))
    
