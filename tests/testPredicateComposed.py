import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import networkx as nx
from   stldec.decomposition_module import *
from   matplotlib.patches import Rectangle
import itertools


# define optimization problem 

opti = ca.Opti()

#########################################################################################################
# Graph creation
#########################################################################################################
# create a set of node positions 

v0 = np.array([-5.,-5.])
v1 = np.array([ 5.,-5.])
v2 = np.array([-5.,0.])
v3 = np.array([ 5.,0.])
v4 = np.array([-5.,5.])
v5 = np.array([ 5.,5.])
v6 = np.array([-10.,7.])
v7 = np.array([ 10.,7.])

# select maximum communication distance
maximumInterRobotDistance  = 50                  # not binding for now
problemDimension           = len(v1)             # dimension of the problem in R^n
numberOfVerticesHypercube  = 2**problemDimension # number of vertices in a hypercube in this dimensional space
maxDistanceConstraint      = maxDistancePredicate(dimension=problemDimension,maxDistance=maximumInterRobotDistance) # obtain predicate for maximum distance constraint



#########################################################################################################
# Predicate construction
#########################################################################################################

## predicates for communicating agents
# set isotropic distance predicate between agent 23 and 45
radius    = 20
predicate = ellipsoidPredicate(np.diag([0.1/radius**2,1/radius**2]),center=np.zeros((problemDimension,1)))
predicateObj = convexPredicateFunction(function=predicate,centerGuess=np.zeros((problemDimension,1)))
print(predicateObj._vertices)