import numpy as np
import matplotlib.pyplot as plt
import os


# Setup directory structure.
current_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = "results"

agentId = [1,2,3,4]

agentsTrajectory = dict()
for id in agentId :
    
    data_file = os.path.join(current_dir, results_dir, f"agent{id}.npy")
    try :
        with open( data_file , 'rb') as f:
            agentsTrajectory[id] = np.load(f)
    except :
        raise Exception("You need to run the simulation before plotting")

# Unpack data and start plotting.
rows,cols = np.shape(agentsTrajectory[1])
time = agentsTrajectory[1][:,2]

t25Index = np.argmin(np.abs(time-25))


fig1,ax1 = plt.subplots()
fig2,ax2 = plt.subplots()

fig1.set_size_inches(w=8,h=8)
fig2.set_size_inches(w=8,h=8)
ax       = [ax1,ax2]


ax[0].grid(visible=True)
ax[1].grid(visible=True)
step1   = 230

color1 = np.array([1,0,1,0.8])
color2 = np.array([0.5,0.5,0,0.8])
color3 = np.array([0,1,1,0.8])
color4 = np.array([1,0,0,0.8])

colors = [color1,color2,color3,color4]
cmaps = dict()
   
for agentId,trajectory in agentsTrajectory.items() :

    x1 = trajectory[:t25Index,0]
    y1 = trajectory[:t25Index,1]

    ax[0].scatter(x1[::step1],y1[::step1],color=colors[agentId-1],linewidths=2)
    ax[0].scatter(x1[0],y1[0],c="green", linewidths=3)
    ax[0].scatter(x1[-1],y1[-1],c="k",marker="x", linewidths=3)

    ax[0].annotate(xy=(x1[0]+0.6,y1[0]+0.6),text=f"agent {agentId}")

ax[0].set_xlabel("x-axis [m]")
ax[0].set_ylabel("y-axis [m]")

step2   = 15

for agentId,trajectory in agentsTrajectory.items() :

    x = trajectory[t25Index:,0]
    y = trajectory[t25Index:,1]
    
    x1 = trajectory[:t25Index,0]
    y1 = trajectory[:t25Index,1]
    

    ax[1].scatter(x[::step2],y[::step2],color=colors[agentId-1],linewidths=2)
    ax[1].scatter(x1[::step1],y1[::step1],color=colors[agentId-1],linewidths=2,alpha=0.2)
    
    ax[1].scatter(x[0],y[0],c="green", linewidths=3)
    ax[1].scatter(x[-1],y[-1],c="k",marker="x", linewidths=3)
    ax[1].annotate(xy=(x[0]+0.6,y[0]+0.6),text=f"agent {agentId}")
    


ax[1].set_xlabel("$x-axis [m]$")
ax[1].set_ylabel("$y-axis [m]$")

ax[0].set_xlim(-12.,9)
ax[0].set_ylim(-17.,12)

ax[1].set_xlim(-4.5,13)
ax[1].set_ylim(-10.,13)

# fig1.savefig('plot1.pdf', format='pdf')
# fig2.savefig('plot2.pdf', format='pdf')

plt.show()