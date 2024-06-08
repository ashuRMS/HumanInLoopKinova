#!/usr/bin/env python3
import rospy,csv,os
from time import time
from sensor_msgs.msg import JointState
from kortex_driver.srv import *
from kortex_driver.msg import *
import casadi as ca
import numpy as np
import kinpy as kp
from casadi import sin, cos, pi
# import matplotlib.pyplot as plt
# from simulation_code import simulate

class CasadiMPC():
    def __init__(self):

        # setting matrix_weights' variables
        self.Q_x = 1
        self.Q_y = 1
        self.Q_z = 1
        self.Q_qw = 0.1
        self.Q_qx = 0.1
        self.Q_qy = 0.1
        self.Q_qz = 0.1

        self.R1 = 1
        self.R2 = 1
        self.R3 = 1
        self.R4 = 1
        self.R5 =1
        self.R6 = 1

        self.step_horizon = 0.05  # time between steps in seconds
        self.N = 10              # number of look ahead steps
        self.sim_time = 200      # simulation time

        # Initial states
        self.x_init =0
        self.y_init = 0
        self.z_init = 0
        self.qw_init = 0
        self.qx_init = 0
        self.qy_init = 0
        self.qz_init =1

        # Final states
        self.x_target = 0.432
        self.y_target = 0.194
        self.z_target = 0.448
        self.qw_target = 0.1763
        self.qx_target = 0.1898
        self.qy_target = 0.685
        self.qz_target =0.6807

        # Control Bounds
        self.v_max = 0.05
        self.v_min = -0.05


        # symobolic variables
        # state symbolic variables
        self.x = ca.SX.sym('x')
        self.y = ca.SX.sym('y')
        self.z = ca.SX.sym('z')
        self.qw = ca.SX.sym('qw')
        self.qx = ca.SX.sym('qx')
        self.qy = ca.SX.sym('qy')
        self.qz = ca.SX.sym('qz')


        self.theta = ca.SX.sym('theta')
        self.states = ca.vertcat(
            self.x,
            self.y,
            self.z,
            self.qw,
            self.qx,
            self.qy,
            self.qz
        )
        self.n_states = self.states.numel()

        # control symbolic variables
        self.V_x = ca.SX.sym('V_x')
        self.V_y = ca.SX.sym('V_y')
        self.V_z = ca.SX.sym('V_z')
        self.W_x = ca.SX.sym('W_x')
        self.W_y = ca.SX.sym('W_y')
        self.W_z = ca.SX.sym('W_z')
        self.controls = ca.vertcat(
            self.V_x,
            self.V_y,
            self.V_z,
            self.W_x,
            self.W_y,
            self.W_z
        )
        self.n_controls = self.controls.numel()



        # matrix containing all states over all time steps +1 (each column is a state vector)
        self.X = ca.SX.sym('X', self.n_states, self.N + 1)

        # matrix containing all control actions over all time steps (each column is an action vector)
        self.U = ca.SX.sym('U', self.n_controls, self.N)

        # coloumn vector for storing initial state and target state
        self.P = ca.SX.sym('P', self.n_states + self.n_states)

        # state weights matrix (Q_X, Q_Y, Q_THETA)
        self.Q = ca.diagcat(self.Q_x, self.Q_y, self.Q_z, self.Q_qw, self.Q_qx, self.Q_qy, self.Q_qz)

        # controls weights matrix
        self.R = ca.diagcat(self.R1, self.R2, self.R3, self.R4, self.R5, self.R6)


        self.t0 = 0
        self.state_init = ca.DM([self.x_init, self.y_init, self.z_init, self.qw_init, self.qx_init, self.qy_init, self.qz_init])        # initial state
        # print("init_state size",self.state_init.shape)
        # exit()
        self.state_target = ca.DM([self.x_target, self.y_target, self.z_target, self.qw_target, self.qx_target, self.qy_target, self.qz_target])  # target state


        # discretization model (e.g. x2 = f(x1, v, t) = x1 + v * dt)
        self.B = ca.vertcat(
            ca.horzcat(1, 0, 0, 0, 0, 0),
            ca.horzcat(0, 1, 0, 0, 0, 0),
            ca.horzcat(0, 0, 1, 0, 0, 0),
            ca.horzcat(0, 0, 0, -self.state_init[4,0], -self.state_init[5,0], -self.state_init[6,0]),
            ca.horzcat(0, 0, 0, self.state_init[3,0], -self.state_init[6,0], self.state_init[5,0]),
            ca.horzcat(0, 0, 0, self.state_init[6,0], self.state_init[3,0], -self.state_init[4,0]),
            ca.horzcat(0, 0, 0, -self.state_init[5,0], self.state_init[4,0], self.state_init[3,0])
        )

        # RHS = states + J @ controls * step_horizon  # Euler discretization
        self.RHS = self.B @ self.controls
        # maps controls from [va, vb, vc, vd].T to [vx, vy, omega].T
        self.f = ca.Function('f', [self.states, self.controls], [self.RHS])


        self.cost_fn = 0  # cost function
        self.g = self.X[:, 0] - self.P[:self.n_states]  # constraints in the equation


        # runge kutta
        for k in range(self.N):
            st = self.X[:, k]
            con = self.U[:, k]
            self.cost_fn = self.cost_fn \
                + (st - self.P[self.n_states:]).T @ self.Q @ (st - self.P[self.n_states:]) \
                + con.T @ self.R @ con
            st_next = self.X[:, k+1]
            k1 = self.f(st, con)
            k2 = self.f(st + self.step_horizon/2*k1, con)
            k3 = self.f(st + self.step_horizon/2*k2, con)
            k4 = self.f(st + self.step_horizon * k3, con)
            st_next_RK4 = st + (self.step_horizon / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            self.g = ca.vertcat(self.g, st_next - st_next_RK4)


        self.OPT_variables = ca.vertcat(
            self.X.reshape((-1, 1)),   # Example: 3x11 ---> 33x1 where 3=states, 11=N+1
            self.U.reshape((-1, 1))
        )
        self.nlp_prob = {
            'f': self.cost_fn,
            'x': self.OPT_variables,
            'g': self.g,
            'p': self.P
        }

        self.opts = {
            'ipopt': {
                'max_iter': 20000,
                'print_level': 0,
                'acceptable_tol': 1e-8,
                'acceptable_obj_change_tol': 1e-6
            },
            'print_time': 0
        }

        self.solver = ca.nlpsol('solver', 'ipopt', self.nlp_prob, self.opts)

        self.lbx = ca.DM.zeros((self.n_states*(self.N+1) + self.n_controls*self.N, 1))
        self.ubx = ca.DM.zeros((self.n_states*(self.N+1) + self.n_controls*self.N, 1))

        self.lbx[0: self.n_states*(self.N+1): self.n_states] = -ca.inf       # X lower bound
        self.lbx[1: self.n_states*(self.N+1): self.n_states] = -ca.inf   # Y lower bound
        self.lbx[2: self.n_states*(self.N+1): self.n_states] =  -ca.inf      # theta lower bound
        self.lbx[3: self.n_states*(self.N+1): self.n_states] = -ca.inf 
        self.lbx[4: self.n_states*(self.N+1): self.n_states] = -ca.inf 
        self.lbx[5: self.n_states*(self.N+1): self.n_states] = -ca.inf 
        self.lbx[6: self.n_states*(self.N+1): self.n_states] = -ca.inf 

        self.ubx[0: self.n_states*(self.N+1): self.n_states] = ca.inf      # X upper bound
        self.ubx[1: self.n_states*(self.N+1): self.n_states] = ca.inf       # Y upper bound
        self.ubx[2: self.n_states*(self.N+1): self.n_states] = ca.inf     # theta upper bound
        self.ubx[3: self.n_states*(self.N+1): self.n_states] = ca.inf 
        self.ubx[4: self.n_states*(self.N+1): self.n_states] = ca.inf 
        self.ubx[5: self.n_states*(self.N+1): self.n_states] = ca.inf 
        self.ubx[6: self.n_states*(self.N+1): self.n_states] = ca.inf                                                                                           

        self.lbx[self.n_states*(self.N+1):] = self.v_min                  # v lower bound for all V
        self.ubx[self.n_states*(self.N+1):] = self.v_max                  # v upper bound for all V


        self.args = {
            'lbg': ca.DM.zeros((self.n_states*(self.N+1), 1)),  # constraints lower bound
            'ubg': ca.DM.zeros((self.n_states*(self.N+1), 1)),  # constraints upper bound
            'lbx': self.lbx,
            'ubx': self.ubx
        }

        

        # xx = DM(state_init)
        self.t = ca.DM(self.t0)

        self.u0 = ca.DM.zeros((self.n_controls, self.N))  # initial control
        self.X0 = ca.repmat(self.state_init, 1, self.N+1)         # initial state full
        

        self.mpc_iter = 0
        self.cat_states = np.array(self.X0.full())
        self.cat_controls = np.array(self.u0[:, 0].full())
        self.times = np.array([[0]])



        # Robot initialization

        rospy.init_node('example_full_arm_movement_python')

        self.HOME_ACTION_IDENTIFIER = 2

        # Get node params
        self.robot_name = rospy.get_param('~robot_name', "gen3_lite") #changed the robot name here
        self.degrees_of_freedom = rospy.get_param("/" + self.robot_name + "/degrees_of_freedom", 6)
        self.is_gripper_present = rospy.get_param("/" + self.robot_name + "/is_gripper_present", True)

        rospy.loginfo("Using robot_name " + self.robot_name + " , robot has " + str(self.degrees_of_freedom) + " degrees of freedom and is_gripper_present is " + str(self.is_gripper_present))

        self.reference_frame_counter=0


        # Load the robot urdf
        with open("/home/ashutoshsahu/catkin_kinova/src/ros_kortex/kortex_description/robots/model.urdf", "r") as urdf_file:
            self.urdf_string = urdf_file.read()

        # build the serial chain from base to tool_frame
        self.serialchain=kp.build_serial_chain_from_urdf(self.urdf_string,"tool_frame")
        self.desired_orientation=np.array([0.17630111, 0.189819 ,  0.68515811, 0.68076797])
        self.current_orientation=None
        self.rate=rospy.Rate(10)
        self.curr_state=None
        # Pubsub
        # Joint state subscriber
        self.sub=rospy.Subscriber('/gen3_lite/joint_states',JointState,self.update_pose)
        self.vel_pub=rospy.Publisher('/gen3_lite/in/cartesian_velocity',TwistCommand,queue_size=10)
        self.curr_joint_angles=None
        

        rospy.loginfo("Using robot_name " + self.robot_name + " , robot has " + str(self.degrees_of_freedom) + " degrees of freedom and is_gripper_present is " + str(self.is_gripper_present))
        rospy.sleep(1.0)
    
    def update_pose(self,data):
        self.curr_joint_angles=data.position[:6]
        # print("updated:", self.curr_joint_angles)
        self.reference_frame_counter+=1
        self.state_init=ca.DM(self.forward_kinematics())
        # print("fk",self.state_init.shape)

    def forward_kinematics(self):
        tool_frame_pose=self.serialchain.forward_kinematics(self.curr_joint_angles)
        position,orient=tool_frame_pose.pos,tool_frame_pose.rot
        state = np.concatenate((position,orient))
        
        return state

    def shift_timestep(self,step_horizon, t0, state_init, u, f):
        f_value = f(state_init, u[:, 0])

        next_state = ca.DM.full(state_init + (step_horizon * f_value))
       
        # print("state: ",next_state)
        # print("shape:",next_state.shape)
        # exit()

        t0 = t0 + step_horizon
        u0 = ca.horzcat(
            u[:, 1:],
            ca.reshape(u[:, -1], -1, 1)
        )

        return t0, self.state_init, u0

    # Function to read a specific line
    def read_specific_line(self,line_to_retrieve):
        file_path='output.csv'
        with open(file_path, mode='r', newline='') as file:
            reader = csv.reader(file)
            # Skip to the desired line
            row = next(islice(reader, line_to_retrieve - 1, line_to_retrieve), None)
            return row



###############################################################################

if __name__ == '__main__':
    main_loop = time()  # return time in sec
    mpc=CasadiMPC()
    # print("1")
    # i=1
    # k=2
    ss_error = ca.norm_2(mpc.state_init - mpc.state_target)
    # tau=0.05
    # File path to save the CSV file
    file_path = 'output.csv'

    # Column headers
    headers = ['Counter', 'Joint1_vel', 'Joint2_vel', 'Joint3_vel', 'Joint4_vel', 'Joint5_vel', 'Joint6_vel']

    # Check if the file already exists to determine if headers need to be written
    file_exists = os.path.isfile(file_path)

    while (ss_error > 0.06) :


        mpc.B = ca.vertcat(
            ca.horzcat(1, 0, 0, 0, 0, 0),
            ca.horzcat(0, 1, 0, 0, 0, 0),
            ca.horzcat(0, 0, 1, 0, 0, 0),
            ca.horzcat(0, 0, 0, -mpc.state_init[4,0], -mpc.state_init[5,0], -mpc.state_init[6,0]),
            ca.horzcat(0, 0, 0, mpc.state_init[3,0], -mpc.state_init[6,0], mpc.state_init[5,0]),
            ca.horzcat(0, 0, 0, mpc.state_init[6,0], mpc.state_init[3,0], -mpc.state_init[4,0]),
            ca.horzcat(0, 0, 0, -mpc.state_init[5,0], mpc.state_init[4,0], mpc.state_init[3,0])
        )

        ss_error = ca.norm_2(mpc.state_init - mpc.state_target)
        t1 = time()
        # print("2")
        mpc.args['p'] = ca.vertcat(
            mpc.state_init,    # current state
            mpc.state_target   # target state
        )
        # optimization variable current state
        mpc.args['x0'] = ca.vertcat(
            ca.reshape(mpc.X0, mpc.n_states*(mpc.N+1), 1),
            ca.reshape(mpc.u0, mpc.n_controls*mpc.N, 1)
        )

        sol = mpc.solver(
            x0=mpc.args['x0'],
            lbx=mpc.args['lbx'],
            ubx=mpc.args['ubx'],
            lbg=mpc.args['lbg'],
            ubg=mpc.args['ubg'],
            p=mpc.args['p']
        )

        mpc.u = ca.reshape(sol['x'][mpc.n_states * (mpc.N + 1):], mpc.n_controls, mpc.N)
        mpc.X0 = ca.reshape(sol['x'][: mpc.n_states * (mpc.N+1)], mpc.n_states, mpc.N+1)

        mpc.cat_states = np.dstack((
            mpc.cat_states,
            np.array(mpc.X0.full())
        ))

        mpc.cat_controls = np.vstack((
            mpc.cat_controls,
            np.array(mpc.u[:, 0].full())
        ))
        mpc.t = np.vstack((
            mpc.t,
            mpc.t0
        ))

        mpc.t0, mpc.state_init, mpc.u0 = mpc.shift_timestep(mpc.step_horizon, mpc.t0, mpc.state_init, mpc.u, mpc.f)

        mpc.X0 = ca.horzcat(
            mpc.X0[:, 1:],
            ca.reshape(mpc.X0[:, -1], -1, 1)
        )

        # xx ...
        t2 = time()
        # print(mpc.mpc_iter)
        print("loop_time: ",t2-t1)
        mpc.times = np.vstack((
            mpc.times,
            t2-t1
        ))

        mpc.mpc_iter = mpc.mpc_iter + 1

        # Velocity command
        vel=TwistCommand()
        # print("B",mpc.B)
        vel.reference_frame=mpc.reference_frame_counter
        # print("current time u:",mpc.u0)
        
        vel.twist.linear_x = mpc.u0[0,0]
        # print(mpc.u0[0,0])
        # exit()
        vel.twist.linear_y = mpc.u0[1,0]
        vel.twist.linear_z = mpc.u0[2,0]
        vel.twist.angular_x = mpc.u0[3,0]
        vel.twist.angular_y = mpc.u0[4,0]
        vel.twist.angular_z = mpc.u0[5,0]
        
    # Velocity command
    vel=TwistCommand()
    vel.reference_frame=mpc.reference_frame_counter
    vel.twist.angular_x = 0
    vel.twist.angular_y = 0
    vel.twist.angular_z = 0
    vel.twist.linear_x = 0
    vel.twist.linear_y = 0
    vel.twist.linear_z = 0
    mpc.vel_pub.publish(vel)
    # print("final error :",np.linalg.norm(quaternion_error))
    # exit()

    main_loop_time = time()
    ss_error = ca.norm_2(mpc.state_init - mpc.state_target)

    # print('\n\n')
    # print('Total time: ', main_loop_time - main_loop)
    # print('avg iteration time: ', np.array(mpc.times).mean() * 1000, 'ms')
    # print('final error: ', ss_error)
    exit()

# print(mpc.state_init)

