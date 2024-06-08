#!/usr/bin/env python3
import rospy
from time import time
from threading import Thread
from sensor_msgs.msg import JointState
from kortex_driver.srv import *
from kortex_driver.msg import *
import casadi as ca
import numpy as np
import kinpy as kp
import csv
import matplotlib.pyplot as plt
from sensor_msgs.msg import Joy,JointState

class CasadiMPC():
    def __init__(self,x,y,z,qw,qx,qy,qz):
        self.initialize_parameters(x,y,z,qw,qx,qy,qz)
        self.initialize_symbols()
        self.initialize_matrices()
        # self.initialize_optimization_problem()
        self.initialize_robot()
    
    def initialize_parameters(self,x,y,z,qw,qx,qy,qz):
        # Set parameters
        self.Q_x = 1
        self.Q_y = 1
        self.Q_z = 1
        self.Q_qw = 1
        self.Q_qx = 1
        self.Q_qy = 1
        self.Q_qz = 1

        self.R1 = .1
        self.R2 = .1
        self.R3 = .1
        self.R4 = .1
        self.R5 =.1
        self.R6 = .1

        self.step_horizon = 0.02
        self.N = 10              
        self.sim_time = 200      

        self.x_init = 0
        self.y_init = 0
        self.z_init = 0
        self.qw_init = 0
        self.qx_init = 0
        self.qy_init = 0
        self.qz_init = 1

        self.x_target = x
        self.y_target = y
        self.z_target = z
        self.qw_target = qw
        self.qx_target = qx
        self.qy_target = qy
        self.qz_target = qz

        self.v_max = 0.05
        self.v_min = -0.05
    def initialize_symbols(self):
        # Define symbolic variables
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

    def initialize_matrices(self):
        # Initialize matrices
        self.X = ca.SX.sym('X', self.n_states, self.N + 1)
        self.U = ca.SX.sym('U', self.n_controls, self.N)
        self.P = ca.SX.sym('P', self.n_states + self.n_states)

        self.Q = ca.diagcat(self.Q_x, self.Q_y, self.Q_z, self.Q_qw, self.Q_qx, self.Q_qy, self.Q_qz)
        self.R = ca.diagcat(self.R1, self.R2, self.R3, self.R4, self.R5, self.R6)

        self.t0 = 0
        self.state_init = ca.DM([self.x_init, self.y_init, self.z_init, self.qw_init, self.qx_init, self.qy_init, self.qz_init])
        self.state_target = ca.DM([self.x_target, self.y_target, self.z_target, self.qw_target, self.qx_target, self.qy_target, self.qz_target])

        self.B = ca.vertcat(
            ca.horzcat(1, 0, 0, 0, 0, 0),
            ca.horzcat(0, 1, 0, 0, 0, 0),
            ca.horzcat(0, 0, 1, 0, 0, 0),
            ca.horzcat(0, 0, 0, -self.state_init[4,0], -self.state_init[5,0], -self.state_init[6,0]),
            ca.horzcat(0, 0, 0, self.state_init[3,0], -self.state_init[6,0], self.state_init[5,0]),
            ca.horzcat(0, 0, 0, self.state_init[6,0], self.state_init[3,0], -self.state_init[4,0]),
            ca.horzcat(0, 0, 0, -self.state_init[5,0], self.state_init[4,0], self.state_init[3,0])
        )

        self.RHS = self.B @ self.controls
        self.f = ca.Function('f', [self.states, self.controls], [self.RHS])

        self.cost_fn = 0
        self.g = self.X[:, 0] - self.P[:self.n_states]

        for k in range(self.N):
            # Runge-Kutta
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
            self.X.reshape((-1, 1)),
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

        self.lbx[0: self.n_states*(self.N+1): self.n_states] = -0.70       # X lower bound
        self.lbx[1: self.n_states*(self.N+1): self.n_states] = -0.70   # Y lower bound
        self.lbx[2: self.n_states*(self.N+1): self.n_states] =  0     # theta lower bound
        self.lbx[3: self.n_states*(self.N+1): self.n_states] = -1
        self.lbx[4: self.n_states*(self.N+1): self.n_states] = -1
        self.lbx[5: self.n_states*(self.N+1): self.n_states] = -1 
        self.lbx[6: self.n_states*(self.N+1): self.n_states] = -1

        self.ubx[0: self.n_states*(self.N+1): self.n_states] = 0.70      # X upper bound
        self.ubx[1: self.n_states*(self.N+1): self.n_states] = 0.70      # Y upper bound
        self.ubx[2: self.n_states*(self.N+1): self.n_states] = 0.70      # theta upper bound
        self.ubx[3: self.n_states*(self.N+1): self.n_states] = 1 
        self.ubx[4: self.n_states*(self.N+1): self.n_states] = 1 
        self.ubx[5: self.n_states*(self.N+1): self.n_states] = 1 
        self.ubx[6: self.n_states*(self.N+1): self.n_states] = 1                                                                                            

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

    def initialize_robot(self):
        rospy.init_node('example_full_arm_movement_python')
        # Initialize robot

        rospy.init_node('example_full_arm_movement_python')

        self.HOME_ACTION_IDENTIFIER = 2

        # Get node params
        self.robot_name = rospy.get_param('~robot_name', "gen3_lite") #changed the robot name here
        self.degrees_of_freedom = rospy.get_param("/" + self.robot_name + "/degrees_of_freedom", 6)
        self.is_gripper_present = rospy.get_param("/" + self.robot_name + "/is_gripper_present", True)

        # Xbox
        self.joy_sub = rospy.Subscriber('/joy', Joy, self.xbox_joy_callback)
        self.reference_frame_counter = 0  # Initialize the reference frame counter
        self.joystick_x = 0
        self.joystick_y = 0
        self.joystick_z = 0
        self.joystick_roll = 0
        self.joystick_pitch = 0
        self.joystick_yaw = 0
        self.rotation_sense = 1
        self.value = 0
        self.gripper_thread = None

        self.alpha=0.1



        # MPC hil task velocity
        self.mpctwist_linear_x = 0
        self.mpctwist_linear_y = 0
        self.mpctwist_linear_z = 0
        self.mpctwist_angular_x = 0
        self.mpctwist_angular_y = 0
        self.mpctwist_angular_z = 0


        # plotting
        self.linear_x_values = []
        self.calculation1lx_values = []
        self.calculation2lx_values = []

        self.linear_y_values = []
        self.calculation1ly_values = []
        self.calculation2ly_values = []

        self.linear_z_values = []
        self.calculation1lz_values = []
        self.calculation2lz_values = []

        self.angular_x_values = []
        self.calculation1ar_values = []
        self.calculation2ar_values = []

        self.angular_y_values = []
        self.calculation1ap_values = []
        self.calculation2ap_values = []

        self.angular_z_values = []
        self.calculation1ay_values = []
        self.calculation2ay_values = []


        self.iteration_values = []


        # Load the robot urdf
        with open("/home/ashutoshsahu/catkin_kinova/src/ros_kortex/kortex_description/robots/model.urdf", "r") as urdf_file:
            self.urdf_string = urdf_file.read()

        # build the serial chain from base to tool_frame
        self.serialchain=kp.build_serial_chain_from_urdf(self.urdf_string,"tool_frame")
        self.desired_orientation=np.array([0.17630111, 0.189819 ,0.68515811, 0.68076797])
        self.current_orientation=None
        self.rate=rospy.Rate(40)
        self.curr_state=None
        # Pubsub
        # Joint state subscriber
        self.sub=rospy.Subscriber('/gen3_lite/joint_states',JointState,self.update_pose)
        self.vel_pub=rospy.Publisher('/gen3_lite/in/cartesian_velocity',TwistCommand,queue_size=10)
        self.curr_joint_angles=None
        

        rospy.loginfo("Using robot_name " + self.robot_name + " , robot has " + str(self.degrees_of_freedom) + " degrees of freedom and is_gripper_present is " + str(self.is_gripper_present))



    def xbox_joy_callback(self, msg):
        self.joystick_x = msg.axes[0]  # Joystick X axis (Translation along X)
        self.joystick_y = msg.axes[1]  # Joystick Y axis (Translation along Y)
        self.joystick_z = msg.axes[3]  # Joystick Z axis (Translation along Z)
        self.joystick_roll = msg.buttons[0]  # Roll axis (Rotation around X) A
        self.joystick_pitch = msg.buttons[1]  # Pitch axis (Rotation around Y) B
        self.joystick_yaw = msg.buttons[3]  # Yaw axis (Rotation around Z) Y
        self.rotation_sense = msg.axes[3]  # PRESS LEFT TRIGGER TO CHANGE THE SENSE OF ROTATION
        self.value = msg.axes[5]

        if self.rotation_sense == 0:
            self.rotation_sense = 1

        if msg.buttons[4]==1:
            self.alpha += 0.02
            # Ensure alpha stays within the range of 0 to 1
            self.alpha = min(self.alpha, 1)
            print(self.alpha)

            # print("changed to false")
        if msg.buttons[5]==1:
            # print("here")
            self.alpha-= 0.06
            self.alpha = max(0.1, self.alpha)
            print(self.alpha)

    def publish_twist_command(self):
        self.reference_frame_counter += 1
        k=0.5
        # Initialize lists to store the values
        

        twist_command = TwistCommand()
        twist_command.reference_frame = self.mpc_iter
        twist_command.twist.linear_x =  0.1*k*self.alpha*self.joystick_x + self.mpctwist_linear_x*(1-self.alpha)
        twist_command.twist.linear_y =  0.1*k* self.joystick_y + self.mpctwist_linear_y*(1-self.alpha)
        twist_command.twist.linear_z = 0.1*k*self.joystick_z + self.mpctwist_linear_z*(1-self.alpha)
        twist_command.twist.angular_x = 0.1*k*self.joystick_roll * self.rotation_sense + self.mpctwist_angular_x*(1-self.alpha)
        twist_command.twist.angular_y = 0.1*k*self.joystick_pitch * self.rotation_sense + self.mpctwist_angular_y*(1-self.alpha)
        twist_command.twist.angular_z =  0.1*k*self.joystick_yaw * self.rotation_sense* + self.mpctwist_angular_z*(1-self.alpha)

        self.vel_pub.publish(twist_command)
        # Collect values for plotting
        self.iteration_values.append(self.mpc_iter)

        # self.linear_x_values.append(0.1*k*self.alpha*self.joystick_x  + self.mpctwist_linear_x*(1-self.alpha) )
        self.calculation1lx_values.append(0.1 *k* self.alpha * self.joystick_x)
        # print(0.1 *k* self.alpha * self.joystick_x)
        # print(self.calculation1lx_values)
        self.calculation2lx_values.append(self.mpctwist_linear_x * (1 - self.alpha))
        

        # self.linear_y_values.append(0.1*k*self.alpha*self.joystick_y  + self.mpctwist_linear_y*(1-self.alpha) )
        self.calculation1ly_values.append(0.1 *k* self.alpha * self.joystick_y)
        self.calculation2ly_values.append(self.mpctwist_linear_y * (1 - self.alpha))
        


        # self.linear_z_values.append(0.1*k*self.alpha*self.joystick_z  + self.mpctwist_linear_z*(1-self.alpha) )
        self.calculation1lz_values.append(0.1 * k*self.alpha * self.joystick_z)
        self.calculation2lz_values.append(self.mpctwist_linear_z * (1 - self.alpha))
        

        # self.angular_x_values.append(0.1*k*self.alpha*self.joystick_roll  + self.mpctwist_angular_x*(1-self.alpha) )
        self.calculation1ar_values.append(0.1 * k*self.alpha * self.joystick_roll)
        self.calculation2ar_values.append(self.mpctwist_angular_x * (1 - self.alpha))
        

        # self.angular_y_values.append(0.1*k*self.alpha*self.joystick_pitch  + self.mpctwist_angular_y*(1-self.alpha) )
        self.calculation1ap_values.append(0.1 *k* self.alpha * self.joystick_pitch)
        self.calculation2ap_values.append(self.mpctwist_angular_y * (1 - self.alpha))
        
            
        # self.angular_z_values.append(0.1*k*self.alpha*self.joystick_yaw  + self.mpctwist_angular_z*(1-self.alpha) )
        self.calculation1ay_values.append(0.1 * k*self.alpha * self.joystick_yaw)
        self.calculation2ay_values.append(self.mpctwist_angular_z * (1 - self.alpha))
    


        

        if self.value == 1:
            if self.gripper_thread is None or not self.gripper_thread.is_alive():
                self.gripper_thread = Thread(target=self.example_send_gripper_command, args=(self.value,))
                self.gripper_thread.start()
        elif self.value == -1:
            if self.gripper_thread is None or not self.gripper_thread.is_alive():
                self.gripper_thread = Thread(target=self.example_send_gripper_command, args=(0,))
                self.gripper_thread.start()

    def example_send_gripper_command(self, value):
        req = SendGripperCommandRequest()
        finger = Finger()
        finger.finger_identifier = 0
        finger.value = value
        req.input.gripper.finger.append(finger)
        req.input.mode = GripperMode.GRIPPER_POSITION

        # rospy.loginfo("Sending the gripper command...")

        try:
            send_gripper_command_full_name = '/gen3_lite/base/send_gripper_command'
            rospy.wait_for_service(send_gripper_command_full_name)
            send_gripper_command = rospy.ServiceProxy(send_gripper_command_full_name, SendGripperCommand)
            send_gripper_command(req)
        except rospy.ServiceException:
            rospy.logerr("Failed to call SendGripperCommand")
    def update_pose(self, data):
        # Update pose
        self.curr_joint_angles=data.position[:6]
        # print("updated:", self.curr_joint_angles)
        self.reference_frame_counter+=1
        self.forward_kinematics()
        # print("fk",self.state_init.shape)

    def forward_kinematics(self):
        # Calculate forward kinematics
        tool_frame_pose=self.serialchain.forward_kinematics(self.curr_joint_angles)
        position,orient=tool_frame_pose.pos,tool_frame_pose.rot
        state = np.concatenate((position,orient))
        self.state_init=ca.DM(state)
        
    def shift_timestep(self, step_horizon, t0, state_init, u, f):
        # Shift timestep
        f_value = f(state_init, u[:, 0])


        t0 = t0 + step_horizon
        u0 = ca.horzcat(
            u[:, 1:],
            ca.reshape(u[:, -1], -1, 1)
        )

        return t0, self.state_init, u0

    



def main(mpc):
    ss_error = ca.norm_2(mpc.state_init - mpc.state_target)
    while ss_error > 0.02:

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

        mpc.mpc_iter = mpc.mpc_iter + 1



        # Velocity command
        vel=TwistCommand()
        # print("B",mpc.B)
        
        mpc.mpctwist_linear_x = mpc.u0[0,1]
        mpc.mpctwist_linear_y = mpc.u0[1,1]
        mpc.mpctwist_linear_z = mpc.u0[2,1]
        mpc.mpctwist_angular_x = mpc.u0[3,1]
        mpc.mpctwist_angular_y = mpc.u0[4,1]
        mpc.mpctwist_angular_z = mpc.u0[5,1]
        mpc.publish_twist_command()
        mpc.rate.sleep()

    # Velocity command for stopping
    vel = TwistCommand()
    vel.reference_frame = mpc.reference_frame_counter
    mpc.mpctwist_linear_x = 0
    mpc.mpctwist_linear_y =0
    mpc.mpctwist_linear_z = 0
    mpc.mpctwist_angular_x = 0
    mpc.mpctwist_angular_y = 0
    mpc.mpctwist_angular_z = 0
    mpc.vel_pub.publish(vel)

    
    from datetime import datetime

    # Generate a unique filename with current date and time
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_file_name = f"values_{current_time}.csv"
    # Save the values to a CSV file
    with open(csv_file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Iteration', 'joyx', 'MPCx', 'joyy', 'MPCy', 'joyz', 'MPCz','joyax', 'MPCax', 'joyay', 'MPCay', 'joyaz', 'MPCaz'])
        for i in range(len(mpc.calculation1lx_values)):
            writer.writerow([i, mpc.calculation1lx_values[i], mpc.calculation2lx_values[i],
            mpc.calculation1ly_values[i], mpc.calculation2ly_values[i],
            mpc.calculation1lz_values[i], mpc.calculation2lz_values[i],
            mpc.calculation1ar_values[i], mpc.calculation2ar_values[i],
            mpc.calculation1ap_values[i], mpc.calculation2ap_values[i],
            mpc.calculation1ay_values[i], mpc.calculation2ay_values[i]])
            
        print("saving file")


    # print('\n\n')
    # print('final error: ', ss_error)
    # exit()

if __name__ == '__main__':
    main_loop = time()
    mpc = CasadiMPC(0.45,0.45,0.420, 0.1765052,0.189839,0.68567,0.6801)
    main(mpc)
    main_loop = time()
    mpc2 = CasadiMPC(0.44,-0.194,0.448,0.17630111, 0.189819 ,0.68515811, 0.68076797)
    print("here")
    main(mpc2)
    
