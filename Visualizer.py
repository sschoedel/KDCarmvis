import ForwardKinematics as FK

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.widgets import Button
from tqdm import tqdm
import time

arm = FK.BarrettArm()
# goal_config1 = [1.46320, 1.76402, -.52058, -0.29301, 0.41901, 0.84979, 0.12817]
# goal_config1 = [0.46320, 1.16402, 2.22058, -0.29301, 0.41901, 0.84979, 0.12817]
goal_config1 = [0.46320, 1.16402, 1.22058, -0.29301, 0.41901, 0.84979, 0.12817]
goal_config2 = [0.49796, 0.98500, 2.34041, -0.11698, 0.07755, 0.82524, 0.54706]

goal_config = goal_config1

g_goal = FK.xyz_quat_to_transform(goal_config1)
axis_length = .1
goal_p = g_goal[:3,-1]
goal_axes = np.array([(g_goal @ np.append(np.eye(3)[:,i]*axis_length, 1))[:-1] for i in range(3)])
print(goal_axes)

# Create 3D plot
plt.style.use('seaborn-whitegrid')

fig = plt.figure(figsize=(5, 6))
ax = fig.add_subplot(projection='3d')
plt.subplots_adjust(bottom=0.35, top=1.0)

maxZ = 2.8
minZ = 2.0
maxX = 0.8
minX = 0
maxY = 1.6
minY = 0.8

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.view_init(azim=30)

# Define visual update functions
def drawEverything(jointPositions):
    global goal_config
    ax.clear()
    ax.scatter([0, 0], [0, 0], [maxZ, minZ], s=0)
    plotPointsLines(jointPositions)
    
    # Draw goal point and coordinate frame
    ax.scatter(goal_config[0], goal_config[1], goal_config[2], color='black')
    ax.plot3D((goal_p[0], goal_axes[0,0]), (goal_p[1], goal_axes[0,1]), (goal_p[2], goal_axes[0,2]), color='red')
    ax.plot3D((goal_p[0], goal_axes[1,0]), (goal_p[1], goal_axes[1,1]), (goal_p[2], goal_axes[1,2]), color='green')
    ax.plot3D((goal_p[0], goal_axes[2,0]), (goal_p[1], goal_axes[2,1]), (goal_p[2], goal_axes[2,2]), color='blue')

    ax.axis([minX, maxX, minY, maxY])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    fig.canvas.draw_idle()

def plotPointsLines(jointPositions):
    jointPositions = np.hstack([np.array([0, 0, 0]).reshape(3,1), jointPositions])
    jointPositions = jointPositions[:,0:] # Specify any number of points to not be printed
    # Plot points
    ax.scatter(jointPositions[0,:], jointPositions[1,:], jointPositions[2,:], color='.3')
    for i, row in enumerate(jointPositions.T):
        if i == 0:
            ax.text(row[0], row[1], row[2], 'S', fontsize=10, color='red')
        elif i == len(jointPositions.T)-1:
            ax.text(row[0], row[1], row[2], 'T', fontsize=10, color='red')
        else:
            if i == 6:
                ax.text(row[0], row[1], row[2], '6', fontsize=10, color='red')
            ax.text(row[0], row[1], row[2], i-1, fontsize=10, color='red')
    # Plot lines
    xlines = np.array(jointPositions[0,:])
    ylines = np.array(jointPositions[1,:])
    zlines = np.array(jointPositions[2,:])
    ax.plot3D(xlines, ylines, zlines, 'chartreuse')
    
# eepos_transform = arm.get_ee_transform([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
# eepos_xyz_quat = arm.get_ee_transform_xyz_quat([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
# Js = arm.get_jacobian([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
start_thetas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
jointPositions = arm.get_joint_positions(start_thetas)
drawEverything(jointPositions)
joint_traj = arm.jacobian_pseudoinverse_iterative(start_thetas, goal_config1)

start_p = arm.get_ee_position(start_thetas)
endpositions = [start_p]
for thetas in joint_traj:
    pos = arm.get_ee_position(thetas)
    endpositions.append(pos.copy())
endpositions = np.array(endpositions)
        
def drawNext(event):
    global index
    thetas = joint_traj[index]
    print(f"thetas: {thetas}")
    jointPositions = arm.get_joint_positions(thetas)
    drawEverything(jointPositions)
    print(f"index: {index}")
    index += 1

def drawTraj(event):
    num_frames = 100
    skip = int(len(joint_traj) / num_frames)
    for i, thetas in enumerate(joint_traj[::skip]):
        jointPositions = arm.get_joint_positions(thetas)
        drawEverything(jointPositions)
        plt.pause(0.001)
    ax.plot3D(endpositions[:i:skip,0], endpositions[:i:skip,1], endpositions[:i:skip,2], color='blue')
        
# Create button to start drawing from txt file with joint angles
startDrawingAx = fig.add_axes([0.25, 0.5, 0.1, 0.06])
startDrawingButton = Button(startDrawingAx, "start")
startDrawingButton.on_clicked(drawTraj)

plt.show()