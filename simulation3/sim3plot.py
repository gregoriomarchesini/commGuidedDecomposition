import numpy as np
import matplotlib.pyplot as plt


agentId = [1,2,3,4,5,6,7,8]

agentsTrajectory = dict()
for id in agentId :
    with open(f"./agent{id}.npy", 'rb') as f:
        agentsTrajectory[id] = np.load(f)


rows,cols = np.shape(agentsTrajectory[1])
time = agentsTrajectory[1][:,2]

t10Index = np.argmin(np.abs(time-10))

t25Index = np.argmin(np.abs(time-25))


fig1,ax1 = plt.subplots()
fig2,ax2 = plt.subplots()

fig1.set_size_inches(w=8,h=8)
fig2.set_size_inches(w=8,h=8)
ax       = [ax1,ax2]


ax[0].grid(visible=True)
ax[1].grid(visible=True)
step1   = 100

color1 = np.array([1  ,0  ,1  ,0.8])
color2 = np.array([0.5,0.5,0  ,0.8])
color3 = np.array([0.3,0.5  ,0.5 ,0.8])
color4 = np.array([1  ,0  ,0  ,0.8])
color5 = np.array([0.4,0.5,0.2,0.8])

color6 = np.array([0.5  ,0.6  ,0.8,0.8])
color7 = np.array([0.8  ,0.6  ,0.2,.8])
color8 = np.array([0.1  ,0.1  ,0.4,0.8])



colors = [color1,color2,color3,color4,color5,color6,color7,color8]
cmaps = dict()
   
for agentId,trajectory in agentsTrajectory.items() :

    x1 = trajectory[:t10Index,0]
    y1 = trajectory[:t10Index,1]

    ax[0].scatter(x1[::step1],y1[::step1],color=colors[agentId-1],linewidths=2)
    ax[0].plot(x1[0],y1[0],c="green",marker="D",markersize = 12)
    ax[0].plot(x1[-1],y1[-1],c="k",marker="*",markersize = 12)

    ax[0].annotate(xy=(x1[0]+0.6,y1[0]+0.6),text=f"agent {agentId}")

ax[0].set_xlabel("$x-axis [m]$")
ax[0].set_ylabel("$y-axis [m]$")

step2   = 120

for agentId,trajectory in agentsTrajectory.items() :

    x = trajectory[t10Index:,0]
    y = trajectory[t10Index:,1]
    
    x1 = trajectory[:t10Index,0]
    y1 = trajectory[:t10Index,1]
    

    ax[1].scatter(x[::step2],y[::step2],color=colors[agentId-1],linewidths=2)
    ax[1].scatter(x1[t25Index::step1],y1[t25Index::step1],color=colors[agentId-1],linewidths=2,alpha=0.2)
    ax[1].plot(x[0],y[0],c="green", marker="D",markersize = 12)
    ax[1].plot(x[-1],y[-1],c="k",marker="*",markersize = 12)
    if agentId ==7 :
        ax[1].annotate(xy=(x[-1]-0.4,y[-1]+1),text=f"agent {agentId}")
    else :
        ax[1].annotate(xy=(x[-1]-0.2,y[-1]+0.8),text=f"agent {agentId}")

    


ax[1].set_xlabel("$x-axis [m]$")
ax[1].set_ylabel("$y-axis [m]$")

ax[0].set_xlim(-35.,35)
ax[0].set_ylim(-35.,35)
ax[0].set_xlim(-35.,35)
ax[0].set_ylim(-35.,35)


fig1.savefig('plot1.pdf', format='pdf')
fig2.savefig('plot2.pdf', format='pdf')

plt.show()