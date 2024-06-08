#!/usr/bin/env python3
import rospy,csv,os
from itertools import islice
from time import time
from sensor_msgs.msg import JointState
from kortex_driver.srv import *
from kortex_driver.msg import *
import casadi as ca
import numpy as np
import kinpy as kp
from casadi import sin, cos, pi


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
        self.N = 10            # number of look ahead steps
        self.sim_time = 200     # simulation time

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
        self.P = ca.SX.sym('P', self.n_states,self.N + 1)

        # state weights matrix (Q_X, Q_Y, Q_THETA)
        self.Q = ca.diagcat(self.Q_x, self.Q_y, self.Q_z, self.Q_qw, self.Q_qx, self.Q_qy, self.Q_qz)

        # controls weights matrix
        self.R = ca.diagcat(self.R1, self.R2, self.R3, self.R4, self.R5, self.R6)


        self.t0 = 0
        self.state_init = ca.DM([self.x_init, self.y_init, self.z_init, self.qw_init, self.qx_init, self.qy_init, self.qz_init])        # initial state
        # print("init_state size",self.state_init.shape)
        # exit()
        self.state_target = ca.DM([self.x_target, self.y_target, self.z_target, self.qw_target, self.qx_target, self.qy_target, self.qz_target])  # target state
        
        self.X[:, 0] = self.P[:,0]

        for k in range(self.N):
            st = self.X[:, k]
            con = self.U[:, k]
            # discretization model (e.g. x2 = f(x1, v, t) = x1 + v * dt)
            self.B = ca.vertcat(
                ca.horzcat(1, 0, 0, 0, 0, 0),
                ca.horzcat(0, 1, 0, 0, 0, 0),
                ca.horzcat(0, 0, 1, 0, 0, 0),
                ca.horzcat(0, 0, 0, -st[4], -st[5], -st[6]),
                ca.horzcat(0, 0, 0, st[3], -st[6], st[5]),
                ca.horzcat(0, 0, 0, st[6], st[3], -st[4]),
                ca.horzcat(0, 0, 0, -st[5], st[4], st[3])
            )
            self.X[:, k + 1] = self.X[:, k] + self.step_horizon*self.B@self.U[:, k]



        self.obj = 0
        self.g = []

        for k in range(self.N):
            st = self.X[:, k]
            con = self.U[:, k]
            self.obj = self.obj + ca.mtimes((st - self.P[:,k+1]).T, ca.mtimes(self.Q, self.P[:,k+1])) + ca.mtimes(con.T, ca.mtimes(self.R, con))
        

        for k in range(self.N + 1):
            self.g = ca.vertcat(self.g, self.X[0, k])
            self.g = ca.vertcat(self.g, self.X[1, k])

        self.OPT_variables = ca.vertcat(
            #Example: 3x11 ---> 33x1 where 3=states, 11=N+1
            self.U.reshape((-1, 1))
        )
        self.nlp_prob = {
            'f': self.obj,
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

        self.lbx = ca.DM.zeros((self.n_controls*self.N, 1))
        self.ubx = ca.DM.zeros((self.n_controls*self.N, 1))                                                                        

        self.lbx[self.n_states*(self.N+1):] = self.v_min                  # v lower bound for all V
        self.ubx[self.n_states*(self.N+1):] = self.v_max                  # v upper bound for all V


        self.args = {
            'lbx': self.lbx,
            'ubx': self.ubx
        }

        

        # xx = DM(state_init)
        self.t = ca.DM(self.t0)

        self.u0 = ca.DM.zeros((self.n_controls, self.N))  # initial control
        self.X0 = ca.repmat(self.state_init, 1, self.N+1)         # initial state full
        # print("X0 shape:",self.X0.shape)

        self.mpc_iter = 0
        self.cat_states = np.array(self.X0.full())
        self.cat_controls = np.array(self.u0[:, 0].full())
        self.times = np.array([[0]])


        # Trajectory initialization
        self.Rp = ca.DM([])

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
        self.desired_orientation=np.array([0.17630111, 0.189819 , 0.68515811, 0.68076797])
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

    def generate_trajectory_ts(self):
        # Parameters
        R = .05 # radius in m
        N =300  # number of steps for trajectory
        T = 10  # total time in seconds
        Cx, Cy = 0.4036, 0.15864 # center of the circle in m, to keep home position at circle trajectory

        # Time step
        delta_t = T / N

        for i in range(N):
            t = i * delta_t
            x = Cx + R * np.cos(2 * np.pi * i / N + np.pi/4)
            y = Cy + R * np.sin(2 * np.pi * i / N + np.pi/4)
            
            rp=[x,y,self.z_target,self.qw_target,self.qx_target,self.qy_target,self.qz_target]
            # Convert rp to CasADi DM column vector
            rp_col = ca.DM(rp).reshape((-1, 1))

            # Append the new column to Rp
            if self.Rp.size1() == 0:
                self.Rp = rp_col
            else:
                self.Rp = ca.horzcat(self.Rp, rp_col)
            self.Rp = ca.horzcat(self.Rp, rp_col)


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
        
        return state.reshape((-1, 1))

    def shift_timestep(self,step_horizon, t0, state_init, u):
        st=ca.DM(state_init)
        con=ca.DM(u[0,:].T)
        # f_value = f(st,con)
        # st = ca.DM.full(st + (T))
        t0=t0+step_horizon

        u0 = np.vstack((u[1:,:],u[-1,:]))
        return t0,state_init,u0
        

    # Function to read a specific line
    def read_specific_line(self,file_path,line_to_retrieve):
        # file_path='output.csv'
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
    i=1
    

    mpc.generate_trajectory_ts()
    ss_error = ca.norm_2(mpc.state_init - mpc.state_target)

    for t in range(mpc.Rp.shape[1]):

        mpc.state_target=ca.DM([mpc.x_target, mpc.y_target, mpc.z_target, mpc.qw_target, mpc.qx_target, mpc.qy_target, mpc.qz_target])
        ss_error = ca.norm_2(mpc.state_init - mpc.state_target)
        # print("error",ss_error)
        # exit()
        while (ss_error > 0.02) :
          

            ss_error = ca.norm_2(mpc.state_init - mpc.state_target)
            t1 = time()
            # print("2")
            mpc.args['p'] = ca.horzcat(
                mpc.state_init,   # current state
                mpc.Rp[:,t:t+mpc.N]   # target state
            )
            # optimization variable current state
            mpc.args['x0'] = ca.vertcat(
                ca.reshape(mpc.u0, mpc.n_controls*mpc.N, 1)
            )
            print(mpc.args['x0'])

            sol = mpc.solver(
                x0=mpc.args['x0'],
                lbx=mpc.args['lbx'],
                ubx=mpc.args['ubx'],
                p=mpc.args['p']
            )
            print("ushape",sol['x'].shape)

            mpc.u = ca.reshape(sol['x'], mpc.n_controls, mpc.N)

            mpc.t = np.vstack((
                mpc.t,
                mpc.t0
            ))

            mpc.t0, mpc.state_init, mpc.u0 = mpc.shift_timestep(mpc.step_horizon, mpc.t0, mpc.state_init, mpc.u)

            t2 = time()
            # print(mpc.mpc_iter)
            # print("loop_time: ",t2-t1)
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
            vel.twist.linear_y = mpc.u0[1,0]
            vel.twist.linear_z = mpc.u0[2,0]
            vel.twist.angular_x = mpc.u0[3,0]
            vel.twist.angular_y = mpc.u0[4,0]
            vel.twist.angular_z = mpc.u0[5,0]
            # data=[
            #         [i,mpc.u0[0,0],mpc.u0[1,0],mpc.u0[2,0],mpc.u0[3,0],mpc.u0[4,0],mpc.u0[5,0]],
            #         [i,mpc.u0[0,1],mpc.u0[1,1],mpc.u0[2,1],mpc.u0[3,1],mpc.u0[4,1],mpc.u0[5,1]],
            #         [i,mpc.u0[0,2],mpc.u0[1,2],mpc.u0[2,2],mpc.u0[3,2],mpc.u0[4,2],mpc.u0[5,2]],
            #         [i,mpc.u0[0,3],mpc.u0[1,3],mpc.u0[2,3],mpc.u0[3,3],mpc.u0[4,3],mpc.u0[5,3]],
            #         [i,mpc.u0[0,4],mpc.u0[1,4],mpc.u0[2,4],mpc.u0[3,4],mpc.u0[4,4],mpc.u0[5,4]],
            #         [i,mpc.u0[0,5],mpc.u0[1,5],mpc.u0[2,5],mpc.u0[3,5],mpc.u0[4,5],mpc.u0[5,5]],
            #         [i,mpc.u0[0,6],mpc.u0[1,6],mpc.u0[2,6],mpc.u0[3,6],mpc.u0[4,6],mpc.u0[5,6]],
            #         [i,mpc.u0[0,7],mpc.u0[1,7],mpc.u0[2,7],mpc.u0[3,7],mpc.u0[4,7],mpc.u0[5,7]],
            #         [i,mpc.u0[0,8],mpc.u0[1,8],mpc.u0[2,8],mpc.u0[3,8],mpc.u0[4,8],mpc.u0[5,8]],
            #         [i,mpc.u0[0,9],mpc.u0[1,9],mpc.u0[2,9],mpc.u0[3,9],mpc.u0[4,9],mpc.u0[5,9]],
                    
            #     ]
            data = [[i] + [mpc.u0[row, col] for row in range(6)] for col in range(mpc.N)]

            # if i <=delay_step:
            vel1=TwistCommand()
            vel1.twist.angular_x = 0
            vel1.twist.angular_y = 0
            vel1.twist.angular_z = 0
            vel1.twist.linear_x = 0
            vel1.twist.linear_y = 0
            vel1.twist.linear_z = 0
            mpc.vel_pub.publish(vel1)

                # # Writing to csv file
                # with open(file_path, mode='a', newline='') as file:
                #     writer = csv.writer(file)

                #     # Write headers only if the file didn't exist before
                #     if not file_exists:
                #         writer.writerow(headers)
                #         file_exists = True
                #         # print("Headers written")

                #     # Write data rows
                #     writer.writerows(data)

                # with open(file_path, mode='a', newline='') as file:
                #     writer = csv.writer(file)

                #     # Write headers only if the file didn't exist before
                #     if not file_exists:
                #         writer.writerow(headers)
                #         file_exists = True
                #         # print("Headers written")

                #     # Write data rows
                #     writer.writerows(data2)
                #     print("Data2 written to file")
                # exit()
            
                        

            # elif i>delay_step:
            #     retrieved_row=mpc.read_specific_line(file_path,k)
            #     # print(type(retrieved_row))
            #     vel2=TwistCommand()
            #     vel2.twist.linear_x = float(retrieved_row[1])
            #     vel2.twist.linear_y = float(retrieved_row[2])
            #     vel2.twist.linear_z = float(retrieved_row[3])
            #     vel2.twist.angular_x = float(retrieved_row[4])
            #     vel2.twist.angular_y = float(retrieved_row[5])
            #     vel2.twist.angular_z = float(retrieved_row[6])
            #     mpc.vel_pub.publish(vel2)
                

            #     # Writing to csv file
            #     with open(file_path, mode='a', newline='') as file:
            #         writer = csv.writer(file)
                    
            #         # Write headers only if the file didn't exist before
            #         if not file_exists:
            #             writer.writerow(headers)
            #             file_exists = True
                    
            #         # Write data rows
            #         writer.writerows(data)
            #         # exit()
            #     k+= delay_step
            i+=1
            mpc.rate.sleep()

        # Finish it
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

    exit()



