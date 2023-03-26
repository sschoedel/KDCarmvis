import numpy as np


def as_skew_symmetric(vec):
    a, b, c = vec
    return np.array([[0, -c, b],
                     [c, 0, -a],
                     [-b, a, 0]])

def as_xi_skew_symmetric(xi_bar):
    v, w = xi_bar[0:3], xi_bar[3:6]
    w_ss = as_skew_symmetric(w)
    res = np.hstack([w_ss, v.reshape(3,1)])
    return np.vstack([res, [0, 0, 0, 1]])
    
def as_exponential(xi, theta):
    v, w = xi[0:3], xi[3:6]
    
    w_ss = as_skew_symmetric(w)
    
    e_w_ss_theta = np.eye(3) + w_ss * np.sin(theta) + w_ss @ w_ss * (1 - np.cos(theta))
    
    top_right = (np.eye(3) - e_w_ss_theta) @ (w_ss @ v) + w.reshape((3,1)) @ w.reshape((1,3)) @ v * theta
    
    threebyfour = np.hstack([e_w_ss_theta, top_right.reshape(3,1)])
    res = np.vstack([threebyfour, [0,0,0,1]])
    return res

def compute_xi_bar(w, q):
    return np.append(-np.cross(w, q), w)

def xyz_quat_to_transform(xyz_quat):
    # Quaternion of the form (q_i, q_j, q_k, q_w)
    xyz = xyz_quat[:3]
    q = xyz_quat[3:]
    transform = np.array([[(1 - 2*q[1]**2 - 2*q[2]**2), (2*q[0]*q[1] - 2*q[3]*q[2]), (2*q[0]*q[2] + 2*q[3]*q[1]), xyz[0]],
                          [(2*q[0]*q[1] + 2*q[3]*q[2]), (1 - 2*q[0]**2 - 2*q[2]**2), (2*q[1]*q[2] - 2*q[3]*q[0]), xyz[1]],
                          [(2*q[0]*q[2] - 2*q[3]*q[1]), (2*q[1]*q[2] + 2*q[3]*q[0]), (1 - 2*q[0]**2 - 2*q[1]**2), xyz[2]],
                          [0, 0, 0, 1]])
    return transform

def inv_transform(transform):
    R = transform[:3,:3]
    p = transform[:3,-1]
    inv_transform = np.eye(4)
    inv_transform[:3,:3] = R.T
    inv_transform[:3,-1] = -R.T @ p
    return inv_transform

class BarrettArm():
    tool_center = np.array([0, 0, 0])
    
    g_s0_0 = np.array([[0, -1, 0, 0.61],
                       [1, 0, 0, 0.72],
                       [0, 0, 1, 1.346],
                       [0, 0, 0, 1]])
    g_s1_0 = np.array([[0, 0, -1, 0.61],
                       [1, 0, 0, 0.72],
                       [0, -1, 0, 1.346],
                       [0, 0, 0, 1]])
    g_s2_0 = np.array([[0, -1, 0, 0.61],
                       [1, 0, 0, 0.72],
                       [0, 0, 1, 1.346],
                       [0, 0, 0, 1]])
    g_s3_0 = np.array([[0, 0, -1, 0.61],
                       [1, 0, 0, 0.72],
                       [0, -1, 0, 1.896],
                       [0, 0, 0, 1]])
    g_s4_0 = np.array([[0, -1, 0, 0.61],
                       [1, 0, 0, 0.72],
                       [0, 0, 1, 1.896],
                       [0, 0, 0, 1]])
    g_s5_0 = np.array([[0, 0, -1, 0.61],
                       [1, 0, 0, 0.72],
                       [0, -1, 0, 2.196],
                       [0, 0, 0, 1]])
    # g_s6_0 = np.array([[0, -1, 0, 0.61],
    #                    [1, 0, 0, 0.72],
    #                    [0, 0, 1, 2.196],
    #                    [0, 0, 0, 1]])
    g_s6_0 = np.array([[0, -1, 0, 0.61],
                       [1, 0, 0, 0.72],
                       [0, 0, 1, 2.196 + 0.06 + 0.12],
                       [0, 0, 0, 1]])

    # g_s7_0 = np.array([[0, -1, 0, 0.61],
    #                    [1, 0, 0, 0.72],
    #                    [0, 0, 1, 2.256 + 0.12],
    #                    [0, 0, 0, 1]])
    
    g_s0 = 0
    g_s1 = 0
    g_s2 = 0
    g_s3 = 0
    g_s4 = 0
    g_s5 = 0
    g_s6 = 0
    # g_s7 = 0
    
    w_0 = [0, 0, 1]
    w_1 = [-1, 0, 0]
    w_2 = [0, 0, 1]
    w_3 = [-1, 0, 0]
    w_4 = [0, 0, 1]
    w_5 = [-1, 0, 0]
    w_6 = [0, 0, 1]
    # w_7 = [0, 0, 1]
    
    q_0 = [0.61, 0.72, 1.346]
    q_1 = [0.61, 0.72, 1.346]
    q_2 = [0.61, 0.72, 1.346]
    q_3 = [0.61, 0.765, 1.896]
    q_4 = [0.61, 0.72, 1.896]
    q_5 = [0.61, 0.72, 2.196]
    q_6 = [0.61, 0.72, 2.196 + 0.06 + 0.12]
    # q_7 = [0.61, 0.72, 2.256]
    
    xi_0 = compute_xi_bar(w_0, q_0)
    xi_1 = compute_xi_bar(w_1, q_1)
    xi_2 = compute_xi_bar(w_2, q_2)
    xi_3 = compute_xi_bar(w_3, q_3)
    xi_4 = compute_xi_bar(w_4, q_4)
    xi_5 = compute_xi_bar(w_5, q_5)
    xi_6 = compute_xi_bar(w_6, q_6)
    # xi_7 = compute_xi_bar(w_7, q_7)
    
    # xi_list = [xi_0, xi_1, xi_2, xi_3, xi_4, xi_5, xi_6, xi_7]
    xi_list = [xi_0, xi_1, xi_2, xi_3, xi_4, xi_5, xi_6]
    
    def __init__(self):
        # np.set_printoptions(precision=4, suppress=True)
        pass
    
    def jacobian_pseudoinverse_iterative(self, start_thetas, goal_config_xyz_quat, use_damped_least_squares=False):
        """
        Compute a trajectory that moves the arm from the start configuration to the goal configuration

        Args:
            start_thetas (list[float]): list of n initial joint angles
            goal_config (list[float]): end effector pose in the form [x, y, z, q_i, q_j, q_k, q_w]
        """
        
        # Convert start thetas to transformation matrix
        g_os = self.get_ee_transform(start_thetas) # transform from origin to start
        
        # Convert goal config to transformation matrix
        # transform from origin to destination
        g_od = xyz_quat_to_transform(goal_config_xyz_quat) # takes quat in form q_i, q_j, q_k, q_w
        
        # Get transform between start and destination        
        g_sd = inv_transform(g_os) @ g_od
        
        pose_diff_start = np.hstack([g_sd[:3,-1], self.rot_to_euler(g_sd[:3,:3])])
        print(f"pose_diff_start: {pose_diff_start}")
        
        # Convert pose_diff_start to origin frame
        R_os = g_os[:3,:3]
        p_os = g_os[:3,-1]
        Adj_os = np.zeros((6,6))
        Adj_os[:3,:3] = R_os # Top left 3x3
        Adj_os[3:,3:] = R_os # Bottom right 3x3
        Adj_os[:3,3:] = as_skew_symmetric(p_os) @ R_os # Top right 3x3
        
        # Initialize loop variables
        pose_diff_origin_frame = Adj_os @ pose_diff_start
        
        joint_trajectory = [np.array(start_thetas)]
        curr_thetas = start_thetas.copy()
        lamb = 0.005
        h = 0.02 # Step size
        tol = 1e-5 # Stop condition tolerance
        converged = False
        i = 0
        max_iters = 10000
        while not converged and i < max_iters:
            J = self.get_jacobian(curr_thetas)
            if use_damped_least_squares:
                delta_thetas = h * J.T @ np.linalg.inv(J @ J.T + lamb**2*np.eye(6)) @ (pose_diff_origin_frame)
            else:
                delta_thetas = h * J.T @ np.linalg.inv(J @ J.T) @ (pose_diff_origin_frame)
            curr_thetas += delta_thetas
            joint_trajectory.append(curr_thetas.copy())
            
            # Convert current thetas to current end effector transform
            g_oc = self.get_ee_transform(curr_thetas)
            
            # Get transform between current (computed in loop) and destination (computed before loop)
            g_cd = inv_transform(g_oc) @ g_od
            
            # Convert g_sd to velocity representation
            pose_diff_curr_frame = np.hstack([g_cd[:3,-1], self.rot_to_euler(g_cd[:3,:3])])
            
            # Convert pose_diff_curr_frame to origin frame
            R_oc = g_oc[:3,:3]
            p_oc = g_oc[:3,-1]
            Adj_oc = np.zeros((6,6))
            Adj_oc[:3,:3] = R_oc # Top left 3x3
            Adj_oc[3:,3:] = R_oc # Bottom right 3x3
            Adj_oc[:3,3:] = as_skew_symmetric(p_oc) @ R_oc # Top right 3x3
            
            pose_diff_origin_frame = Adj_oc @ pose_diff_curr_frame
            
            # Check stopping conditions
            if np.linalg.norm(pose_diff_origin_frame) < tol:
                converged = True
            
            print(f"pose_diff: {pose_diff_origin_frame}, norm(pd): {np.linalg.norm(pose_diff_origin_frame)}, norm(dist): {np.linalg.norm(goal_config_xyz_quat[:3] - g_oc[:3,-1])}")
            i += 1
            
            
        return joint_trajectory
        
    def get_jacobian(self, thetas):
        """
        Compute 6xn Jacobian for current joint configuration.
        The 6 element output is of the form V^s_{st}, which is the column vector
        [v_s; ω_s] where v_s = linear velocities and ω_s = rotational velocities
        of the center of the tool frame in spatial frame coords.
        
        Args:
            thetas (list[float]): n element list of angles for each joint

        Returns:
            np.array: 6xn spatial manipulator Jacobian J^s_{st}
        """
        # Update exponential twist compositions for each joint so we can convert each xi to xi' (xip (prime))
        e_matrices = [as_exponential(xi, theta) for xi, theta in zip(self.xi_list, thetas)]
        es_composed = [0]*len(self.xi_list)
        es_composed[0] = np.eye(4)
        es_composed[1] = e_matrices[0]
        es_composed[2] = e_matrices[0] @ e_matrices[1]
        es_composed[3] = e_matrices[0] @ e_matrices[1] @ e_matrices[2]
        es_composed[4] = e_matrices[0] @ e_matrices[1] @ e_matrices[2] @ e_matrices[3]
        es_composed[5] = e_matrices[0] @ e_matrices[1] @ e_matrices[2] @ e_matrices[3] @ e_matrices[4]
        es_composed[6] = e_matrices[0] @ e_matrices[1] @ e_matrices[2] @ e_matrices[3] @ e_matrices[4] @ e_matrices[5]
       
        self.compute_all_joint_poses(thetas)
        
        xi_ps = [0]*len(self.xi_list)
        
        Adjoint_mats = [np.zeros((6,6)) for i in range(len(self.xi_list))]
        for i, e_composed in enumerate(es_composed):
            R = e_composed[:3,:3]
            p = e_composed[:3,-1]
            Adjoint = Adjoint_mats[i]
            Adjoint[:3,:3] = R # Top left 3x3
            Adjoint[3:,3:] = R # Bottom right 3x3
            Adjoint[:3,3:] = as_skew_symmetric(p) @ R # Top right 3x3
        
            xi_ps[i] = Adjoint @ self.xi_list[i]
        
        Js = np.vstack(xi_ps).T
        return Js
    
    def get_ee_position(self, thetas):
        e_matrices = [as_exponential(xi, theta) for xi, theta in zip(self.xi_list, thetas)]
        res = e_matrices[0] @ e_matrices[1] @ e_matrices[2] @ e_matrices[3] @ e_matrices[4] @ e_matrices[5] @ e_matrices[6] @ self.g_s6_0 @ np.append(self.tool_center, 1)
        # res = e_matrices[0] @ e_matrices[1] @ e_matrices[2] @ e_matrices[3] @ e_matrices[4] @ e_matrices[5] @ e_matrices[6] @ e_matrices[7] @ self.g_s7_0 @ np.append(self.tool_center, 1)
        return res[:-1]

    def quat_to_euler(self, q):
        # Quaternion of the form (q_w, q_i, q_j, q_k)
        # Euler in form Body 3-2-1 (ϕ, θ, ψ)
        phi = np.arctan2(2*(q[0]*q[1] + q[2]*q[3]), 1-2*(q[1]**2 + q[2]**2))
        theta = -np.pi/2 + 2*np.arctan2(np.sqrt(1 + 2*(q[0]*q[2] - q[1]*q[3])), np.sqrt(1 - 2*(q[0]*q[2] - q[1]*q[3])))
        psi = np.arctan2(2*(q[0]*q[3] + q[1]*q[2]), 1-2*(q[2]**2 + q[3]**2))
        return np.array([phi, theta, psi])
    
    def rot_to_quat(self, R):
        # Returns quaternion in form i, j, k, w
        qw = np.sqrt(1 + R[0,0] + R[1,1] + R[2,2])/2
        qx = (R[2,1] - R[1,2]) / (4*qw)
        qy = (R[0,2] - R[2,0]) / (4*qw)
        qz = (R[1,0] - R[0,1]) / (4*qw)
        return np.array([qx, qy, qz, qw])

    def rot_to_euler(self, R):
        q = self.rot_to_quat(R) # returns quaternion in form i, j, k, w
        euler = self.quat_to_euler([q[3], q[0], q[1], q[2]]) # requires quaternion in form w, i, j, k
        return euler
        

    def get_ee_transform_xyz_quat(self, thetas):
        # Returns quaternion in form i, j, k, w
        e_matrices = [as_exponential(xi, theta) for xi, theta in zip(self.xi_list, thetas)]
        res = e_matrices[0] @ e_matrices[1] @ e_matrices[2] @ e_matrices[3] @ e_matrices[4] @ e_matrices[5] @ e_matrices[6] @ self.g_s6_0
        # res = e_matrices[0] @ e_matrices[1] @ e_matrices[2] @ e_matrices[3] @ e_matrices[4] @ e_matrices[5] @ e_matrices[6] @ e_matrices[7] @ self.g_s7_0
        xyz =  (res @ np.append(self.tool_center, 1))[:-1]
        R = res[:3,:3]
        quat = self.rot_to_quat(R)
        return np.hstack([xyz, quat])
    
    def get_ee_transform_xyz_euler(self, thetas):
        # Returns thetas in form ϕ, θ, ψ
        xyz_quat = self.get_ee_transform_xyz_quat(thetas)
        pos = np.array(xyz_quat[:3])
        quat = xyz_quat[3:] # x y z w
        euler = self.quat_to_euler([quat[3], quat[0], quat[1], quat[2]]) # quat_to_euler takes w x y z
        xyz_euler = np.hstack([pos, euler])
        return xyz_euler
        

    def get_ee_transform(self, thetas):
        thetas = np.append(thetas, 0)
        e_matrices = [as_exponential(xi, theta) for xi, theta in zip(self.xi_list, thetas)]
        transform = e_matrices[0] @ e_matrices[1] @ e_matrices[2] @ e_matrices[3] @ e_matrices[4] @ e_matrices[5] @ e_matrices[6] @ self.g_s6_0
        # transform = e_matrices[0] @ e_matrices[1] @ e_matrices[2] @ e_matrices[3] @ e_matrices[4] @ e_matrices[5] @ e_matrices[6] @ e_matrices[7] @ self.g_s7_0
        transform[:,-1] = transform @ np.append(self.tool_center, 1)
        return transform
    
    def get_joint_positions(self, thetas):
        '''
        thetas: list or np.array of scalars with theta_0 first and theta_6 last
        
        returns: 3x8 matrix of column vectors corresponding to each joint 
                    position's (x,y,z) in the spatial/global frame
        '''
        self.compute_all_joint_poses(thetas)
        
        origin_homo = np.array([0, 0, 0, 1])
        
        # Create matrix where each column vector corresponds to an [x, y, z, 1] joint position
        joint_positions_homo = np.vstack([self.g_s0 @ origin_homo,
                                          self.g_s1 @ origin_homo,
                                          self.g_s2 @ origin_homo,
                                          self.g_s3 @ origin_homo,
                                          self.g_s4 @ origin_homo,
                                          self.g_s5 @ origin_homo,
                                          self.g_s6 @ origin_homo]).T
                                        #   self.g_s6 @ np.append(self.tool_center, 1)]).T
                                        #   self.g_s7 @ origin_homo,
                                        #   self.g_s7 @ np.append(self.tool_center, 1)]).T
        return joint_positions_homo[:-1,:]

        
    def compute_all_joint_poses(self, thetas):
        '''
        thetas: list or np.array of scalars with theta_0 first and theta_6 last
        '''
        # thetas = np.append(thetas, 0) # Add theta = 0 for joint 7
        e_matrices = [as_exponential(xi, theta) for xi, theta in zip(self.xi_list, thetas)]
        self.g_s0 = e_matrices[0] @ self.g_s0_0
        self.g_s1 = e_matrices[0] @ e_matrices[1] @ self.g_s1_0
        self.g_s2 = e_matrices[0] @ e_matrices[1] @ e_matrices[2] @ self.g_s2_0
        self.g_s3 = e_matrices[0] @ e_matrices[1] @ e_matrices[2] @ e_matrices[3] @ self.g_s3_0
        self.g_s4 = e_matrices[0] @ e_matrices[1] @ e_matrices[2] @ e_matrices[3] @ e_matrices[4] @ self.g_s4_0
        self.g_s5 = e_matrices[0] @ e_matrices[1] @ e_matrices[2] @ e_matrices[3] @ e_matrices[4] @ e_matrices[5] @ self.g_s5_0
        self.g_s6 = e_matrices[0] @ e_matrices[1] @ e_matrices[2] @ e_matrices[3] @ e_matrices[4] @ e_matrices[5] @ e_matrices[6] @ self.g_s6_0
        # self.g_s7 = e_matrices[0] @ e_matrices[1] @ e_matrices[2] @ e_matrices[3] @ e_matrices[4] @ e_matrices[5] @ e_matrices[6] @ e_matrices[7] @ self.g_s7_0
    
    def __str__(self):
        print("Arm xi column vectors:")
        print("--------------------------------------------")
        for i, xi in enumerate(self.xi_list):
            print(f"xi_{i}: {xi}")
        return ""