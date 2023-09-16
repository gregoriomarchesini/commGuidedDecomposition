import numpy as np
import casadi as ca
import itertools
from   typing import Self
from   predicate_builder_module import *
import networkx as nx 
import matplotlib.pyplot as plt 

# TODO:set computation of optimal under approximation only if the fomula is called inside the cycle closure. Otherwise don't set it so you save some computational power

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
        

def edgeSet(path:list,isCycle:bool=False) -> list[(int,int)] :
    """Given a list of nodes, it returns the edges of the path as (n,n+1) 
    Inputs 
    ------------------------------------------------------------------------------------------------
    path (list<float>) : list of nodes in the path
    
    Output 
    ------------------------------------------------------------------------------------------------
    edges (list<tuple>): list of tuples (n,n+1)
    """

    if not isCycle :
      edges = [(path[i],path[i+1]) for i in range(len(path)-1)]
    elif isCycle : # due to how networkx returns edges
      edges = [(path[i],path[i+1]) for i in range(-1,len(path)-1)]
        
    
    return edges


        
class GraphEdge( ) :
    """class GraphEdge 
    This class is useful to define attributes of the edges like STL predicate function, weights ...
    
    class attributes 
    centerVar (cvx.) : edge cvx variable
    nu      (cvx.Variable) : nu   cvx variable for hypercube stateSpaceDimensions
    predicateFunction(function) : function wrapper for predicate function
    """
    
    def __init__(self,source :int,target:int,isCommunicating :int= 0,weight:float=1) -> None:
     
      """ 
      Input
      ----------------------------------
      weight          (float)  : weight of the edge for shortest path algorithms
      isCommunicating (boolean) : 0 is a communicating edge, 1 is a communicating edge
      
      """
     
      if weight<=0 : # only accept positive weights
          raise("Edge weight must be positive")
      
      self._isCommunicating          = isCommunicating
      self._isInvolvedInOptimization = 0
      
      if not(self._isCommunicating) :
          self._weight = float("inf")
      else :
          self._weight = weight
          
      self._formulasList = []  
      
      if (not isinstance(source,int)) or (not isinstance(target,int)) :
          raise ValueError("Target source pairs must be integers")
      else :
          self._sourceNode = source
          self._targetNode = target
      
      
          
    @property
    def formulasList(self) :
        return self._formulasList
    
    @property
    def isCommunicating(self) :
      return self._isCommunicating 
   
    @property
    def sourceNode(self) :
        return self._sourceNode
    @property
    def targetNode(self) :
        return self._targetNode
  
    @property
    def isInvolvedInOptimization(self) :
      return self._isInvolvedInOptimization
  
    @property
    def weight(self):
        return self._weight
    @property
    def hasSpecifications(self):
        return bool(len(self._formulasList)) 

    
    @weight.setter
    def weight(self,new_weight:float)-> None :
        if not isinstance(new_weight,float) :
            raise TypeError("Weight must be a float")
        elif new_weight<0 :
            raise ValueError("Weight must be positive")
        else :
            self._weight = new_weight
            
    
    def addFormula(self,formulas:list|STLformula) -> None :
        """ Set the formulas for the edge that has to be respected by the edge. Input is expected to be a list  """
        
        # this is the case you only have one fomrula
        if not isinstance(formulas,list) :
            if isinstance(formulas,STLformula) :
                # set the source node pairs of this node
                formulas.predicate.addSourceTarget(source=self._sourceNode,target=self._targetNode)
                self._formulasList.append(formulas) # adding a single formula
            else :
                raise Exception("please enter a valid STL formula object or a a list of STLformula objects")
                
        else :
            for formula in formulas :
                if not isinstance(formula,STLformula) :
                    raise Exception("Some of the given formulas are not STLformula objects. please revise your input")
                else :
                    formula.predicate.addSourceTarget(source=self._sourceNode,target=self._targetNode)
                    self._formulasList.append(formula)
                    
    def flagOptimizationInvolvement(self) -> None :
        self._isInvolvedInOptimization = 1
        

########################################################################################################################### 
# FORMULAS CHECKERS
###########################################################################################################################
def allAlways(formulas :list[STLformula]) :
    if len(formulas) :
        for formula in formulas :
            if formula.temporalOperator != "always" :
                return False
        return True 
    else : # in case of zero length
        return False
    
def allDisjoint( formulasList : list[STLformula]) -> bool:
    
    if len(formulasList)==1:
      return True
    if len(formulasList) == 0 :
        raise ValueError("Empty list not accepted")
  
    for i,formulai in enumerate(formulasList) :
        for j,formulaj  in enumerate(formulasList) :
            itervalIntersection = formulai.timeInterval / formulaj.timeInterval
            if (j!= i) and  (not itervalIntersection.isEmpty()) :
                return False
    
    return True   
    
def isThereMultipleIntersection(formulasList : list[STLformula]) -> bool :
    """check if one always has more than one intersection with other always operators"""
   
    if len(formulasList) == 0 :
        raise ValueError("Empty list not accepted")
    for i,formulai in enumerate(formulasList) :
        count = 0
        for j,formulaj  in enumerate(formulasList) :
            
            intervalsIntersection = formulai.timeInterval / formulaj.timeInterval
            if (j!= i) and (not intervalsIntersection.isEmpty()) : # if there is an intersection
                count += 1
                if count>1 :
                    return True
        
    return False
    
def allParametric(listOfFormulas : list[STLformula]) -> bool :
    
    if len(listOfFormulas) == 0 :
        raise ValueError("Empty list not accepted")
    for formula in listOfFormulas :
        if not formula.isParametric :
            return False
    return True

def allNonParametric(listOfFormulas : list[STLformula]) -> bool :
    
    if len(listOfFormulas) == 0 :
        raise ValueError("Empty list not accepted")
    for formula in listOfFormulas :
        if formula.isParametric :
            return False
    return True 


def haveSameTimeInterval(listOfFormulas : list[STLformula]) -> bool :
    """check that all formulas might have the same time interval"""
    if len(listOfFormulas) == 0 :
        raise ValueError("Empty list not accepted")
    if len(listOfFormulas) == 1 :
        return True
    else :
        intervalChecker = listOfFormulas[0].timeInterval
        for formula in listOfFormulas[1:] :
            if formula.timeInterval!=intervalChecker :
                return False
        return True 

def computeTimeIntervalIntersection(formulas :list[STLformula]) -> timeInterval :
    """compute intersection of a list of formulas"""
    # time interval intersection can be computed in sequence 
    
    if len(formulas) <2 :
        raise ValueError("At least two formulas must be given to compute the interval intersection")
    
    intersection = formulas[0].timeInterval / formulas[1].timeInterval
    for formula in formulas[2:] :
        intersection = intersection / formula.timeInterval
    
    return intersection       
     
# this will be changed my man
def computeWeights(source,taget,attributesDict) :
    """takes the edge object from the attributes and returns the wight stored in there"""
    return attributesDict["edgeObj"].weight    

def computeVolume(vector):
    """simply computes the product of elements in a vector"""
    prod = 1
    n,m = vector.shape
    length = max(n,m)
    for jj in range(length):
        prod = prod*vector[jj]
        
    return prod
           

########################################################################################################################### 
# CONSTRAINT COMPUTATION
###########################################################################################################################

# Compute constraints for overloading of functions     
def computeOverloadingConstraints(edgeObj: GraphEdge) -> list[ca.Function]:
    """
    Given a list of formulas defined on a single edge, it computes the required inclusion constraints due to the conjunction of such formulas along the same edge"""
    
    formulas = edgeObj.formulasList
    print("source traget pair",edgeObj.sourceNode,edgeObj.targetNode)
    
    if len(formulas) == 1 :
        return  [] # with one single formula you don't need overloading constraints
    
    # Main assumption check
    alwaysFormulas :list[STLformula]= [formula for formula in formulas if formula.temporalOperator=="always"]
    eventuallyFormulas : list[STLformula] = [formula for formula in formulas if formula.temporalOperator=="eventually"]
 
    constraints = []
    
    sameTimeInterval = False
    if isThereMultipleIntersection(formulasList=alwaysFormulas) :
        if not haveSameTimeInterval(listOfFormulas=alwaysFormulas) : # if there are multiple intersections then they must have the same time interval
            raise NotImplementedError("Seems like there is at least a triple of always formulas that are intersecting in terms of time interval. This can be handled only if all the always operatrs have the same time interval. This is not the case for the formulas you inserted. Possible constrasting inclusion constraints would arise")
        else : # in the case they all have the same time interval divide in parameteric and not parametric and start the inclusion process 
            sameTimeInterval = True # flag so that computation does not have to be redone
        
            
    # PART 1 : resolve always vs always formulas intersections
    if sameTimeInterval : # all the always formulas have the same time interval
        sharedTimeInterval = alwaysFormulas[0].timeInterval
        
        if allParametric(alwaysFormulas) :
            #in case the formulas are all parameteric then we can include them in a sequence of formulas.
            # Each formula includes its successor
            for index in range(len(alwaysFormulas)-1) :
                parentFormula = alwaysFormulas[index]
                childFormula  = alwaysFormulas[index+1]
                
                print("parent formulas ID :", id(parentFormula))
                print("child  formulas ID :", id(childFormula))
                constraints  += parentFormula.getConstraintFromInclusionOf(childFormula) #linear inclusion among two parameteric formulas
            
            print("we set all the inclusions")
            
        else : # case in which the always formulas are non intersecting
            parametricFormulas    = [formula for formula in alwaysFormulas if formula.isParametric]
            nonParametricFormulas = [formula for formula in alwaysFormulas if not formula.isParametric]
            
            if len(nonParametricFormulas)>1 :
                raise NotImplementedError("Seems like there is one edge that constains two always specifications with same time interval. This is not smart. Indeed two always formulas with same time interval require that the superleevl sets are intersecting. So just rewrite the formula using the interscetion of the two orginal superlevel sets as predicate superlevel set")
            
            else :
                alwaysFormulas = nonParametricFormulas + parametricFormulas # we reorder the always formulas such that the non parameteric formula is at the beginning. SO the sequence of inclusions is now correct
                
                for index in range(len(alwaysFormulas)-1) :
                    parentFormula = alwaysFormulas[index]
                    childFormula  = alwaysFormulas[index+1]
                    constraints  += parentFormula.getConstraintFromInclusionOf(childFormula)
        
        innerMostAlwaysFormula = alwaysFormulas[-1] # the always formula the is the last one to receive an inclusion constraint. All the other formulas include this one
            
        if len(eventuallyFormulas) :
            for eventuallyFormula in eventuallyFormulas :
                if eventuallyFormula.timeInterval<=sharedTimeInterval: # always formulas have same
                    constraints  += innerMostAlwaysFormula.getConstraintFromInclusionOf(eventuallyFormula)
                    
    else : # in the case that they don't have the same time interval and the always operators intersections are fixed
        
        if len(alwaysFormulas) >=2 : 
            combinations   = itertools.combinations(range(len(alwaysFormulas)),2) # all possible combinations of always formulas

            # check which combinations have intersections of time intervals and act accordingly
            for combo in combinations :
                formulaI:STLformula       = formulas[combo[0]]
                formulaJ:STLformula       = formulas[combo[1]]
                intersection:timeInterval = formulaI.timeInterval / formulaJ.timeInterval
                
                if not intersection.isEmpty() :
                    
                    if formulaJ.isParametric and not(formulaI.isParametric) : # parametric vs non-parameteric formula
                        constraints  += formulaI.getConstraintFromInclusionOf(formulaJ)
                        
                    elif formulaI.isParametric and not(formulaJ.isParametric) :  
                        constraints  += formulaJ.getConstraintFromInclusionOf(formulaI)
                    
                    elif formulaI.isParametric and formulaJ.isParametric : # if both parameteric include the one with shortest time into the one with longest time (this is our design choice)
                        
                        if formulaI.timeInterval.measure  >=  formulaJ.timeInterval.measure: # then include the formula J in I
                            constraints  += formulaI.getConstraintFromInclusionOf(formulaJ)
                        else :
                            constraints  += formulaJ.getConstraintFromInclusionOf(formulaI)
    
                    else : # if both formulas are non-parameteric then we assume that the disgn of the initial specification was such that it was possible to satisfy them in the first place
                        pass    
        
        
        if len(eventuallyFormulas) :
            for eventuallyFormula in eventuallyFormulas :
                for alwaysFormula in alwaysFormulas :

                    if eventuallyFormula.timeInterval<=alwaysFormula.timeInterval: # if the time interval of the always formulas includes the time interval of the eventually formula
                        if alwaysFormula.isParametric :
                            constraints  += alwaysFormula.getConstraintFromInclusionOf(eventuallyFormula)
                            break # since the always are only intersecting, an eventually can be strictly included only in of of them. 
                                # At most, there is another intersection with another always, but in this case there is no need for forcing the intersection.
                                # In addition, if all the always have the same time interval, the we would eliminate redundant constraints by only putting the 
                                # inclusion on one always since the always are also aready included into each other
                         
    return constraints
                

# compute constraints for cycles of specifications      
def createCycleClosureConstraint(cycleEdgeObjs : list[GraphEdge],cycleEdges : list[(int,int)]) -> list[ca.MX] :
    
    constraints = []
    if len(cycleEdgeObjs) :
        return []
    
    cycleFormulas        : list[list[STLformula]] = [edge.formulasList for edge in cycleEdgeObjs]
    possibleCombinations : list[list[STLformula]] = itertools.product(*cycleFormulas) # all possible combinations of formulas closing a cycle 
    stateSpaceDim        = cycleFormulas[0][0].stateSpaceDimension # first formula that you find
                    
    for combination in possibleCombinations :
        # each combination represent a combinaton of formulas that can possibly close the cycle
        alwaysFormulasInCombination = [formula for formula in combination if formula.temporalOperator == "always" ]
        eventuallyFormulasInCombination = [formula for formula in combination if formula.temporalOperator == "eventually" ]
        
             
        # Case 1: all always formulas 
        if len(combination) == len(alwaysFormulasInCombination) and len(alwaysFormulasInCombination)!=0:
            intervalIntersection : timeInterval = computeTimeIntervalIntersection(alwaysFormulasInCombination)
            
            if not intervalIntersection.isEmpty() : # if there is an intersection then add the constraints 
                constraints += computeMinkowskiInclusionConstraintsForCycle(combination,stateSpaceDim,cycleEdges)
        
        # Case 2:  case in which there are also eventually formulas 
        else :
            if len(eventuallyFormulasInCombination)  ==1 :
                eventuallyFormula    = eventuallyFormulasInCombination[0]
                intervalIntersection = computeTimeIntervalIntersection(alwaysFormulasInCombination)
                # check if the eventually is included in intervalIntersection
                if eventuallyFormula.timeInterval <= intervalIntersection :
                    constraints +=  computeMinkowskiInclusionConstraintsForCycle(combination,stateSpaceDim,cycleEdges)
                    
            elif len(eventuallyFormulasInCombination)  >1 :    
                # now there is more than one eventually formula so try to check if all the eventually formulas have a single instant as time inteval : like [t,t] in the time interval
            
                requiresConstraint = True
                intervalChecker = eventuallyFormulasInCombination[0].timeInterval 
                for eventuallyFormula in eventuallyFormulasInCombination[1:] :  
                    if not eventuallyFormula.timeInterval.isSingular():
                        requiresConstraint = False
                        break # in case even only one eventually formula has non-singular time interval (singular time interval means like [t,t]) then you don't need to add constraints
                    else :
                        if intervalChecker != eventuallyFormula.timeInterval :  # in case the singular time interval is not the same as all the others then you don't need a constraint
                            requiresConstraint = False
                            break
             
                if requiresConstraint :
                    constraints +=  computeMinkowskiInclusionConstraintsForCycle(combination,stateSpaceDim,cycleEdges)
            else :
                pass
              
    return constraints



def computeMinkowskiInclusionConstraintsForCycle(cycleFormulas:list[STLformula],stateSpaceDimension : int, cycleEdges:list[(int,int)] ) -> list[ca.MX]:
    
    # separate formulas
    parametricFormulas    = [formula for formula in cycleFormulas if formula.isParametric]
    parametericEdges      = [edge for edge,formula in zip(cycleEdges,cycleFormulas) if formula.isParametric]
    nonParametricFormulas = [formula for formula in cycleFormulas if not formula.isParametric]
    nonParametericEdges   = [edge for edge,formula in zip(cycleEdges,cycleFormulas) if not formula.isParametric]
    
    constraints = []
    numVertices = 2**stateSpaceDimension
    
    
    if len(nonParametricFormulas) == 0 : # if all the formulas are parametric 
        # in case we select the mid point of the cycle (which must have at least 3 formulas because a minimum cycle is a triangle) 
        midIndex      = int(len(parametricFormulas)/2)
        leftFormulas  = parametricFormulas[0:midIndex]
        rightFormulas = parametricFormulas[midIndex:]
        leftPathEdges      = cycleEdges[0:midIndex]
        rightPathEdges     = cycleEdges[midIndex:]
        
        # compute respective Minkowski sum
        leftVertices     = pathMinkowskiSumVertices(leftFormulas,leftPathEdges)
        A,b              = minkowskiSumLinearRepresentation(rightFormulas,rightPathEdges)
        
        # now we are ready to add the inclusion constraint
        for jj in range(numVertices) :
            constraints += [A@(-leftVertices[:,jj])-b<=np.zeros((2*stateSpaceDimension))] # note the minus sign it is because you need the negative MinkowskySum to be included in the original minkowsky sum
            
        
    elif len(nonParametricFormulas)!=0 and len(parametricFormulas)!=0 : # if there are parametric formulas then you need to compute the Minkowski sum for them vased on ana good approximation for example (here there is a lot of space for improvement)
    # include non parametrif formulas into parameteric formulas
    
        parametericVertices = pathMinkowskiSumVertices(parametricFormulas,parametericEdges)
        # here you actually change the orginal predicate such and you approximate with a cuboid from inside
        for formula in nonParametricFormulas:
            formula.predicate.replaceWithApproximatePredicate() # you need to make the replacement of the originaal predicate with its cuboid under-approximation
            
        A,b = minkowskiSumLinearRepresentation(nonParametricFormulas,nonParametericEdges)
        # now we are ready to add the inclusion constraint
        for jj in range(numVertices) :
            constraints += [A@(-parametericVertices[:,jj])-b<=np.zeros((2*stateSpaceDimension))] # the minus sign in front of the vertex is not a mistake. It is because you want the negative of the minkowski sum
    else : # in case the fomrulas are all non-parameteric we assume that their circle closure is already satisfied when the specification is given
        pass  
    
    return constraints


def minkowskiSumLinearRepresentation(listOfFormulas : list[STLformula],edgeList:list[(int,int)]) -> tuple[ca.MX,ca.MX] :
    """
    Given a list of formulas and the corresponding edges directions of path over which the formulas are defined, we  
    compute the matrix A and b representing the minkowski sum of the formulas along the given path 
    such that the minkowsky sum can be represneted as the linear inequality Ax<=b
    
    """
    
    stateSpaceDimension  = listOfFormulas[0].stateSpaceDimension
    center     = 0 # center of the hypercube
    nuSum      = 0 # dimensions of the hypercube
    

    if allParametric(listOfFormulas=listOfFormulas):
        
        for formula,edgeTuple in zip(listOfFormulas,edgeList) :
    
            if formula.edgeTuple != edgeTuple and formula.edgeTuple!= (edgeTuple[1],edgeTuple[0]) :
                raise ValueError("edge along the path is not maching the corresponding formula. Make sure that the edges order and the formulas order is correct")
            
            elif formula.edgeTuple != edgeTuple : # case in which the directions are not matching
                center = center - formula.centerVar
                nuSum  = nuSum  + formula.nuVar
            else : # case in which the directions are matching
                center = center + formula.centerVar 
                nuSum  = nuSum  + formula.nuVar
        
        A  = np.vstack((np.eye(stateSpaceDimension),-np.eye(stateSpaceDimension)))  # (face normals x hypercube stateSpaceDimension)
        Ac = A@center
        d  = ca.vertcat(nuSum/2,nuSum/2)
        b  = Ac+d
        return A,b
    
    elif allNonParametric(listOfFormulas=listOfFormulas) :
    
        for formula,edgeTuple in zip(listOfFormulas,edgeList) :

            if formula.edgeTuple != edgeTuple and formula.edgeTuple!= (edgeTuple[1],edgeTuple[0]) :
                raise ValueError("edge along the path is not maching the corresponding formula. Make sure that the edges order and the formulas order is correct")
            
            # since this formulas are nonparameteric we need to use the best cuboid under approximation for them
            elif formula.edgeTuple != edgeTuple : # case in which the directions are not matching
                center = center - formula.predicate.optimalApproximationCenter
                nuSum  = nuSum  + formula.predicate.optimalApproximationNu
            else : # case in which the directions are matching
                center = center + formula.predicate.optimalApproximationCenter 
                nuSum  = nuSum  + formula.predicate.optimalApproximationNu
            
        A  = np.vstack((np.eye(stateSpaceDimension),-np.eye(stateSpaceDimension)))  # (face normals x hypercube stateSpaceDimension)
        Ac = A@center
        d  = ca.vertcat(nuSum/2,nuSum/2)
        b  = Ac+d
        return A,b
    else :
        raise Exception("Minkowski sum can be computed only for all parameteric formulas or all non-parameteric formulas. Mixed types of minkowsky sums are not implementd for now")
        
def pathMinkowskiSumVertices(listOfFormulas : list[STLformula],edgeList: list[(int,int)]) -> ca.MX  :
    """
    Given a set of edges with assigned formulas, it computes the vertices of the minkowski sum for the superlevel sets of the given formulas.
    Each formula is define alon one edge of the edgeList in the same sequence as the formulas are given. The edge information is used to 
    decide in which verse we should take the formulas if their direction of definiton is different from the direction of the path for example
    
       NOTE: Only Parametric formulas accepted for this function. So do not use it for other scopes
    
    Inputs
    --------------------------------------------------
    listOfFormulas (list<STLFormula>)
    
    Outputs
    -----------------------------------------
    minkowshySumVertices (cvxpy.Variable): returns a matrix where each column is a vertex of the Minkowski sum of the hypercubes defined by each edge in the list of edges
    """
    if not allParametric(listOfFormulas=listOfFormulas) :
        raise ValueError("Only lists of parametric formulas are accepted")
    elif len(listOfFormulas) != len(edgeList) :
        raise ValueError("list of formulas must have the same length of edges list")
    
    stateSpaceDimension   = listOfFormulas[0].stateSpaceDimension
    cartesianProductSets  = [[-1,1],]*stateSpaceDimension
    hypercubeVertices     = np.array(list(itertools.product(*cartesianProductSets))).T # All vertices of hypercube centered at the origin (Each vertex is a column)
    
    center = 0 # center of the hypercube
    nuSum  = 0 # dimensions of the hypercube
    
    for formula,edgeTuple in zip(listOfFormulas,edgeList) :
        
        if formula.edgeTuple != edgeTuple and formula.edgeTuple!= (edgeTuple[1],edgeTuple[0]) :
            raise ValueError("edge along the path is not maching the corresponding formula. Make sure that the edges order and the formulas order is correct")
        
        elif formula.edgeTuple != edgeTuple : # case in which the directions are not matching
            print("I entered the right case")
            center = center - formula.centerVar
            nuSum  = nuSum  + formula.nuVar
        else : # case in which the directions are matching
            center = center + formula.centerVar 
            nuSum  = nuSum  + formula.nuVar
        
    minkowshySumVertices = center + hypercubeVertices*nuSum/2 # find final hypercube dimension
    
    return minkowshySumVertices
      


def computeNewTaskGraph(MASgraph:nx.Graph,problemDimension:int,maxDistanceFunction : ca.Function)-> tuple[nx.Graph,nx.Graph,nx.Graph] : 
    """ Solves the task decomposition completely"""
    
    numberOfVerticesHypercube = 2**problemDimension
    
    edges = MASgraph.edges(data = True)
    initialTaskGraph   = MASgraph.copy()
    noTaskEdges        = [(i,j) for i,j,attr in edges if not attr["edgeObj"].hasSpecifications] 
    initialTaskGraph.remove_edges_from(noTaskEdges)

    commGraph = MASgraph.copy()
    noCommunication =  [(i,j) for i,j,attr in edges if not attr["edgeObj"].isCommunicating] # because I have rewritten those specification
    commGraph.remove_edges_from(noCommunication)

    pathsList                   : list[list[int]] = []
    pathConstraints             : list[ca.MX] = []
    overloadingConstraints      : list[ca.MX] = []
    cyclesConstraints           : list[ca.MX] = []
    maxCommunicationConstraints : list[ca.MX] = []
    positiveNuConstraint        : list[ca.MX] = []


    for nodei,nodej,attribute in MASgraph.edges(data=True) :
        edgeObject:GraphEdge = attribute["edgeObj"]

        if (not edgeObject.isCommunicating) and (edgeObject.hasSpecifications) : # find a path
            
            # retrive all the formulas to be decomposed
            formulasToBeDecomposed: list[STLformula] = MASgraph.edges[nodei,nodej]["edgeObj"].formulasList 
        
            # path finding and grouping nodes
            path = nx.shortest_path(MASgraph,source=nodei,target=nodej,weight=computeWeights) # path of agents from start to end
            pathsList.append(path) # save sources list for later plotting
            edgesThroughPath = edgeSet(path=path) # find edges along the path
            
            # flag the edges applied for the optimization 
            for sourceNode,targetNode in   edgesThroughPath :
                    MASgraph.edges[sourceNode,targetNode]["edgeObj"].flagOptimizationInvolvement()
            
            # for each formula to be decomposed we will have n subformulas with n being the length of the path we select.
            for formula in formulasToBeDecomposed : # add a new set of formulas for each edge
                edgeSubformulas : list[STLformula] = [] # list of subformulas associate to one orginal formula. you have as many subformulas as number of edges
                
                originalTemporalOperator :str                     = formula.temporalOperator  # get time interval of the orginal operator
                originalTimeInterval     :timeInterval            = formula.timeInterval      # get time interval of the orginal operator
                originalPredicate        :convexPredicateFunction = formula.predicate # get the predicate function
                
                for sourceNode,targetNode in  edgesThroughPath :
                    
                    # create a new parameteric subformula object
                    subformula = STLformula(timeinterval     = originalTimeInterval,
                                            temporalOperator = originalTemporalOperator,
                                            predicate        = convexPredicateFunction(targetNode=targetNode,sourceNode=sourceNode,stateSpaceDimension=problemDimension))
                    
                    # warm start of the variables involved in the optimization TODO: check if you have a better warm start base on the specification you have. Maybe some more intelligen heuristic
                    globalOptimizer.set_initial(subformula.centerVar , MASgraph.nodes[targetNode]["pos"]-MASgraph.nodes[sourceNode]["pos"]) 
                    globalOptimizer.set_initial(subformula.nuVar   , np.ones(problemDimension))

                    # add subformulas to the current path
                    edgeSubformulas.append(subformula)
                    subformulaVertices = subformula.getHypercubeVertices(sourceNode=sourceNode,targetNode=targetNode)
                    
                    # set positivity of dimensions vector nu
                    positiveNuConstraint.append(-np.eye(problemDimension)@subformula.nuVar<=np.zeros((problemDimension,1))) # constraint on positivity of the dimension variable
                    MASgraph.edges[sourceNode,targetNode]["edgeObj"].addFormula(subformula)   # add current subformula to the edge 

                    # Set maximum distance constraint for each hypercube vertex
                    for jj in range(numberOfVerticesHypercube) : 
                        maxCommunicationConstraints.append(maxDistanceFunction(subformulaVertices[:,jj])<=0)     
                        
                
                # now set that the final i sum has to stay inside the orginal predicate
                minowkySumVertices  = pathMinkowskiSumVertices( edgeSubformulas ,  edgesThroughPath)  # return the symbolic vertices f the hypercube to define the constraints
                for jj in range(numberOfVerticesHypercube) :
                        pathConstraints.append(originalPredicate.function(minowkySumVertices[:,jj])<=0) # for each vertex of the minkowski sum ensure they are inside the original predicate superlevel-set
    
    # Now we check the cycle constraints on the graph as a first step and we then check the overloading constraints as a second step
    TaskGraph       = MASgraph.copy()
    noTaskEdges     = [(i,j) for i,j,attr in edges if not attr["edgeObj"].hasSpecifications] 
    noCommunication = [(i,j) for i,j,attr in edges if not attr["edgeObj"].isCommunicating] # because I have rewritten those specification
    TaskGraph.remove_edges_from(noTaskEdges)
    TaskGraph.remove_edges_from(noCommunication)
    
    # adding cycles constraints to the optimization problem
    cycles :list[list[int]]   = sorted(nx.simple_cycles(TaskGraph))

    for omega in cycles :
        cycleEdges    = edgeSet(omega,isCycle=True)
        cycleEdgesObj :list[list[GraphEdge]] = [MASgraph.edges[i,j]["edgeObj"] for i,j in cycleEdges ] 
        cyclesConstraints += createCycleClosureConstraint(cycleEdgeObjs=cycleEdgesObj,cycleEdges=cycleEdges)
        
    # now we compute the overloading constraints on a single objects
    # one line of overloading constraints
    optimisedEdges = [(i,j,edgeDict["edgeObj"]) for i,j,edgeDict in MASgraph.edges(data=True) if edgeDict["edgeObj"].isInvolvedInOptimization]
    for i,j,edgeObj in optimisedEdges  :
        overloadingConstraints += computeOverloadingConstraints(edgeObj)
        
                
    # #########################################################################################################
    # # OPTIMIZATION
    # #########################################################################################################

    cost = 0 # compute cost for parameetric formulas
    for i,j,edgeObj in optimisedEdges :
        for formula in edgeObj.formulasList :
            if formula.isParametric :
                cost = cost + 1/computeVolume(formula.nuVar)
            
    constraints = [*maxCommunicationConstraints,*positiveNuConstraint,*pathConstraints,*cyclesConstraints,*overloadingConstraints]
    globalOptimizer.subject_to(constraints) # Maximum Distance of a constraint


    globalOptimizer.solver("ipopt")
    solution = globalOptimizer.solve()


    # ###########################################################################################################
    # # PRINT SOLUTION
    # #########################################################################################################

    newFormulasCount = 0
    print()
    for i,j,edgeObject in optimisedEdges :
        newFormulasCount += len([formula for formula in edgeObject.formulasList if formula.isParametric])
    
    
    print("-----------------------------------------")   
    print("Internal Report")   
    print("-----------------------------------------")   
    print(f"Total number of formulas created : {newFormulasCount}")   
    print("---------Found Solution------------------") 
    for i,j,edgeObject in optimisedEdges :   
        for formula in edgeObject.formulasList:
            if formula.isParametric :
                print("edge      : (" + str(i)+ "," + str(j) + ")")
                print("vector    : "+ str(solution.value(formula.centerVar)))
                print("dimension : "+ str(solution.value(formula.nuVar)))
                print("formua ID :" + str(id(formula)))
                # turn predicates from parameteric to no parameteric
                
                
    return initialTaskGraph,TaskGraph,commGraph 


def visualizeGraphs(communicationGraph:nx.Graph, initialTaskGraph:nx.Graph, finalTaskGraph:nx.Graph) :
    
    nodes = communicationGraph.nodes(data=True)
    xx = [node[1]["pos"][0] for node in nodes]
    yy = [node[1]["pos"][1] for node in nodes]
    xxmin,xxmax= min(xx)*1.6,max(xx)*1.6
    yymin,yymax = min(yy)*1.6,max(yy)*1.6

    nodes = communicationGraph.nodes(data=True)
    
    # define figure object
    fig, ax = plt.subplots() 
    ax.set_xlim([-20,20])
    ax.set_ylim([-20,20])


    # Define graphs
    fig, ax = plt.subplots(1,3) 

    # Communicatino Graph
    ax[0].set_xlim([xxmin,xxmax])
    ax[0].set_ylim([yymin,yymax])
    # # Drawing of the network
    edgeLabels = { (i,j):"link" for  i,j in communicationGraph.edges}    
    nx.draw_networkx(communicationGraph,{node:nodeDict["pos"] for node,nodeDict in nodes},ax=ax[0])

    nx.draw_networkx_edge_labels(
        communicationGraph,
        {node:nodeDict["pos"] for node,nodeDict in nodes},
        edge_labels = edgeLabels,
        font_color='black',
        ax=ax[0]
    )

    ax[0].set_title("communication graph")


    # final Task Graph
    ax[1].set_xlim([-20,20])
    ax[1].set_ylim([-20,20])
    # # Drawing of the network
    edgeLabels = { (i,j):"Task" for  i,j,attr in finalTaskGraph.edges(data=True) if attr["edgeObj"].hasSpecifications}    
    taskPlot = nx.draw_networkx(finalTaskGraph,{node:nodeDict["pos"] for node,nodeDict in nodes},ax=ax[1])

    nx.draw_networkx_edge_labels(
        finalTaskGraph,
        {node:nodeDict["pos"] for node,nodeDict in nodes},
        edge_labels = edgeLabels,
        font_color='black',
        ax=ax[1]
    )

    ax[1].set_title("final Task Graph")
    
    ax[1].set_xlim([xxmin,xxmax])
    ax[1].set_ylim([yymin,yymax])
    
    
    # Initial Task Graph
    ax[2].set_xlim([-20,20])
    ax[2].set_ylim([-20,20])
    # # Drawing of the network
    edgeLabels = { (i,j):"Task" for  i,j,attr in initialTaskGraph.edges(data=True) if attr["edgeObj"].hasSpecifications}    
    taskPlot = nx.draw_networkx(initialTaskGraph,{node:nodeDict["pos"] for node,nodeDict in nodes},ax=ax[2])

    nx.draw_networkx_edge_labels(
        initialTaskGraph,
        {node:nodeDict["pos"] for node,nodeDict in nodes},
        edge_labels = edgeLabels,
        font_color='black',
        ax=ax[2]
    )

    ax[2].set_title("Initial Task Graph")
    ax[2].set_xlim([xxmin,xxmax])
    ax[2].set_ylim([yymin,yymax])




if __name__== "__main__" :
    # here we run some tests 
    import predicate_builder_module as predmod
     
    # intersecting formulas should give a true
    formula1 = STLformula(stateSpaceDimension=2,temporalOperator="always",timeinterval=timeInterval(10,14))
    formula2 = STLformula(stateSpaceDimension=2,temporalOperator="always",timeinterval=timeInterval(3,11))
    formula3 = STLformula(stateSpaceDimension=2,temporalOperator="always",timeinterval=timeInterval(8,11))
    formulaList = [formula1,formula2,formula3]
    assert isThereMultipleIntersection(formulasList=formulaList)==True, f"formulas are always with intersecting time intervals. Got that there is not intersection"
    assert allParametric(formulaList) == True, f"formulas in the list are all parameteric but got that they are not"
    assert allNonParametric(formulaList)==False, f"formulas in the list are not all non-parameteric but got that they are"
    print("Test 1 passed")
    
    
    formula1 = STLformula(stateSpaceDimension=2,temporalOperator="always",timeinterval=timeInterval(10,14))
    formula2 = STLformula(stateSpaceDimension=2,temporalOperator="always",timeinterval=timeInterval(3,8))
    formula3 = STLformula(stateSpaceDimension=2,temporalOperator="always",timeinterval=timeInterval(15,30))
    formulaList = [formula1,formula2,formula3]
    
    # oveloading constraints test function (For this setting of formulas there should not be any problem)
    try :
        cc = computeOverloadingConstraints(formulaList)
    except :
        raise Exception(" There was an error during this call which should not arise because all the time intervals are non intersecting")
    print("Test 2 passed")
    
    # always formulas with all the same time interval
    formula1 = STLformula(stateSpaceDimension=2,temporalOperator="always",timeinterval=timeInterval(10,12))
    formula2 = STLformula(stateSpaceDimension=2,temporalOperator="always",timeinterval=timeInterval(10,12))
    formula3 = STLformula(stateSpaceDimension=2,temporalOperator="always",timeinterval=timeInterval(10,12))
    formulaList = [formula1,formula2,formula3]
    
    cc = computeOverloadingConstraints(formulaList)
    assert len(cc) == 8,f"There should be exactly 8 constraints for this formula as you have 2 parameteric formulas that are included into a third one. IN a @D case you have 4 verticess constraint for each inclusion. So 2 x 4 = 8"
    print("Test 3 passed")
    
    # always formulas with all the same time interval and a non parameteric formula
    formula1 = STLformula(stateSpaceDimension=2,temporalOperator="always",timeinterval=timeInterval(10,12))
    formula2 = STLformula(stateSpaceDimension=2,temporalOperator="always",timeinterval=timeInterval(10,12))
    formula3 = STLformula(stateSpaceDimension=2,temporalOperator="always",timeinterval=timeInterval(10,12))
    circularPredicate = predmod.maxDistancePredicate(stateSpaceDimension=2,maxDistance=5)
    formula4 = STLformula(stateSpaceDimension=2,temporalOperator="always",timeinterval=timeInterval(10,12),predicate=circularPredicate)
    formulaList = [formula1,formula2,formula3,formula4]
    cc = computeOverloadingConstraints(formulaList)
    
    assert len(cc) == 12,f"There should be exactly 12 constraints for this formula as you have 3 parameteric formulas and 1 non parameteric formula that are included into a third one. IN a @D case you have 4 verticess constraint for each inclusion. So 2 x 4 = 8"
    print("Test 4 passed")
    
    # always formulas with all the same time interval and a non parameteric formula
    formula1 = STLformula(stateSpaceDimension=2,temporalOperator="always",timeinterval=timeInterval(10,12))
    formula2 = STLformula(stateSpaceDimension=2,temporalOperator="always",timeinterval=timeInterval(10,12))
    formula3 = STLformula(stateSpaceDimension=2,temporalOperator="always",timeinterval=timeInterval(10,12))

    formula4 = STLformula(stateSpaceDimension=2,temporalOperator="always",timeinterval=timeInterval(10,12),predicate=predmod.maxDistancePredicate(stateSpaceDimension=2,maxDistance=5))
    formula5 = STLformula(stateSpaceDimension=2,temporalOperator="always",timeinterval=timeInterval(10,12),predicate=predmod.maxDistancePredicate(stateSpaceDimension=2,maxDistance=5))
    formulaList = [formula1,formula2,formula3,formula4,formula5]
    try :
      cc = computeOverloadingConstraints(formulaList)
    except :
        print("Error was rised correctly for this test")
        print("Test 5 passed")
        
    # always formulas with all the same time interval and a non parameteric formula
    formula1 = STLformula(stateSpaceDimension=2,temporalOperator="eventually",timeinterval=timeInterval(11,12))
    formula2 = STLformula(stateSpaceDimension=2,temporalOperator="always",timeinterval=timeInterval(10,12))
    formula3 = STLformula(stateSpaceDimension=2,temporalOperator="always",timeinterval=timeInterval(10,12))
    formula4 = STLformula(stateSpaceDimension=2,temporalOperator="always",timeinterval=timeInterval(10,12),predicate=predmod.maxDistancePredicate(stateSpaceDimension=2,maxDistance=5))
    formulaList = [formula1,formula2,formula3,formula4]
    cc = computeOverloadingConstraints(formulaList)
    assert len(cc)==12, f"Two parameteric always get inserted int the non parameteric one. In addition the eventually gets inserted into one of the alwyas formulas only once"
    
    formula1 = STLformula(stateSpaceDimension=2,temporalOperator="eventually",timeinterval=timeInterval(11,12))
    formula2 = STLformula(stateSpaceDimension=2,temporalOperator="always",timeinterval=timeInterval(10,12))
    formula3 = STLformula(stateSpaceDimension=2,temporalOperator="always",timeinterval=timeInterval(10,12))
    formulaList = [formula1,formula2,formula3]
    
    
    A,b = minkowskiSumLinearRepresentation(formulaList)
    
    
    
    
    vv = pathMinkowskiSumVertices(formulaList)
    
    formulaList = [STLformula(stateSpaceDimension=2,temporalOperator="always",timeinterval=timeInterval(11,12)) for jj in range(5)]
   
    constraints = computeMinkowskiInclusionConstraintsForCycle(cycleFormulas=formulaList,stateSpaceDimension=2)
    assert len(constraints) == 4,f"there should be only 4 active constraints from this condition. Indeed the constraint resolves to a single inclusion of one cuboid into another one. So there is one constraint for each vertex (4 vertices in 2 dimensions) "
    
    
    formulaList1 = [STLformula(stateSpaceDimension=2,temporalOperator="always",timeinterval=timeInterval(11,12)) for jj in range(5)]
    formulaList2 = [STLformula(stateSpaceDimension=2,temporalOperator="eventually",timeinterval=timeInterval(11.4,11.4)) for jj in range(5)]
    formulaList  = formulaList1 + formulaList2
    constraints  = computeMinkowskiInclusionConstraintsForCycle(formulaList,stateSpaceDimension=2)
    
    
    
    
