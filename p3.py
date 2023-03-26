import ForwardKinematics as FK

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np

import time

np.set_printoptions(precision=5, suppress=True)

arm = FK.BarrettArm()
start_thetas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

# Print Jacobian at start config
jac = arm.get_jacobian([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
print(f"Jacobian: \n{jac}")

goal_config1 = [0.46320, 1.16402, 2.22058, -0.29301, 0.41901, 0.84979, 0.12817]
goal_config2 = [0.49796, 0.98500, 2.34041, -0.11698, 0.07755, 0.82524, 0.54706]

goal_config = goal_config2

start = time.time()
joint_traj = arm.jacobian_pseudoinverse_iterative(start_thetas, goal_config, use_damped_least_squares=True)
print(f"total time taken = {time.time() - start}")

start_p = arm.get_ee_position(start_thetas)
endpositions = [start_p]
for thetas in joint_traj:
    pos = arm.get_ee_position(thetas)
    endpositions.append(pos.copy())
endpositions = np.array(endpositions)


# Plot solution
fig = plt.figure(figsize=(5, 6))
ax = fig.add_subplot(projection='3d')
ax.scatter(goal_config[0], goal_config[1], goal_config[2], color='blue')
ax.scatter(start_p[0], start_p[1], start_p[2], color='orange')
ax.plot3D(endpositions[:,0], endpositions[:,1], endpositions[:,2], 'chartreuse')


# Compute goal coordinate frame axes
axis_length = .01
g_goal = FK.xyz_quat_to_transform(goal_config)
goal_p = g_goal[:3,-1]
goal_axes = np.array([(g_goal @ np.append(np.eye(3)[:,i]*axis_length, 1))[:-1] for i in range(3)])

# Draw coordinate frames
ax.plot3D((goal_p[0], goal_axes[0,0]), (goal_p[1], goal_axes[0,1]), (goal_p[2], goal_axes[0,2]), color='red')
ax.plot3D((goal_p[0], goal_axes[1,0]), (goal_p[1], goal_axes[1,1]), (goal_p[2], goal_axes[1,2]), color='green')
ax.plot3D((goal_p[0], goal_axes[2,0]), (goal_p[1], goal_axes[2,1]), (goal_p[2], goal_axes[2,2]), color='blue')

# Compute start coordinate frame axes
g_start = arm.get_ee_transform(joint_traj[0])
start_s = g_start[:3,-1]
start_axes = np.array([(g_start @ np.append(np.eye(3)[:,i]*axis_length, 1))[:-1] for i in range(3)])

# Draw coordinate frames
ax.plot3D((start_p[0], start_axes[0,0]), (start_p[1], start_axes[0,1]), (start_p[2], start_axes[0,2]), color='red')
ax.plot3D((start_p[0], start_axes[1,0]), (start_p[1], start_axes[1,1]), (start_p[2], start_axes[1,2]), color='green')
ax.plot3D((start_p[0], start_axes[2,0]), (start_p[1], start_axes[2,1]), (start_p[2], start_axes[2,2]), color='blue')

# set axis limits
maxZ = 2.25
minZ = maxZ - 0.02
maxX = 0.55
minX = maxX - 0.12
maxY = 1.12
minY = maxY - 0.12

ax.scatter([0, 0], [0, 0], [maxZ, minZ], s=0)
ax.axis([minX, maxX, minY, maxY])

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

ax.legend(["start pose", "goal pose 2"])
plt.show()

# Save joint trajectory file
np.savetxt("joint_trajectory_2_damped.txt", np.array(joint_traj))