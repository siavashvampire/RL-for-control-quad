import time

from app.models.environment.LearnPathEnv import LearnPathEnv
import numpy as np

TIME_SCALING = 1.0  # Any positive number(Smaller is faster). 1.0->Real Time, 0.0->Run as fast as possible
QUAD_DYNAMICS_UPDATE = 0.002  # seconds
CONTROLLER_DYNAMICS_UPDATE = 0.005  # seconds

QUAD_PARAMETERS = {'position': [1, 1, 1], 'orientation': [0, 0, 0], 'L': 0.3, 'r': 0.1, 'prop_size': [10, 4.5],
                   'weight': 1.2}

CONTROLLER_PARAMETERS = {'Motor_limits': [4000, 9000],
                         'Tilt_limits': [-10, 10],
                         'Yaw_Control_Limits': [-900, 900],
                         'Z_XY_offset': 500,
                         'Linear_PID': {'P': [300, 300, 7000], 'I': [0.04, 0.04, 4.5], 'D': [450, 450, 5000]},
                         'Linear_To_Angular_Scaler': [1, 1, 0],
                         'Yaw_Rate_Scaler': 0.18,
                         'Angular_PID': {'P': [22000, 22000, 1500], 'I': [0, 0, 1.2], 'D': [12000, 12000, 0]},
                         }

class TestEnv(LearnPathEnv):
    def __init__(self, ip_address, image_shape, env_config):
        self.eps_n = 0
        super(TestEnv, self).__init__(ip_address, image_shape, env_config)
        self.agent_traveled = []
        self.random_start = False

    def setup_flight(self):
        self.quad.stop_thread()
        self.ctrl.stop_thread()
        time.sleep(1)
        super(TestEnv, self).setup_flight()
        time.sleep(1)

        self.quad.start_thread(dt=QUAD_DYNAMICS_UPDATE, time_scaling=TIME_SCALING)
        self.ctrl.start_thread(update_rate=CONTROLLER_DYNAMICS_UPDATE, time_scaling=TIME_SCALING)
        self.eps_n += 1

        # Start the agent at a random yz position
        # y_pos, z_pos = (0, 0)
        # pose = airsim.Pose(airsim.Vector3r(self.agent_start_pos, y_pos, z_pos))
        # self.drone.simSetVehiclePose(pose=pose, ignore_collision=True)

    def step(self, action):
        self.do_action(action)
        obs, info = self.get_obs()
        reward, done = self.compute_reward()

        position = self.quad.get_position()

        position[0] = round(position[0], 1)
        position[1] = round(position[1], 1)
        position[2] = round(2.5 - position[2], 1)
        position[2] = -0.4
        pose = airsim.Pose(airsim.Vector3r(position[0], position[1], position[2]))
        self.drone.simSetVehiclePose(pose=pose, ignore_collision=True)
        time.sleep(2)
        x, y, z = self.drone.simGetVehiclePose().position
        print("pos = ", z, "asd = ", position[2])

        return obs, reward, done, info

    def compute_reward(self):
        reward = 0
        done = 0

        x, _, _ = self.drone.simGetVehiclePose().position

        if self.is_collision():
            done = 1
            self.agent_traveled.append(x)

        if done and self.eps_n % 5 == 0:
            print("---------------------------------")
            print("> Total episodes:", self.eps_n)
            print("> Flight distance (mean): %.2f" % (np.mean(self.agent_traveled)))
            print("> Holes reached (max):", int(np.max(self.agent_traveled) // 4))
            print("> Holes reached (mean):", int(np.mean(self.agent_traveled) // 4))
            print("---------------------------------\n")

        return reward, done

    def do_action(self, select_action):
        speed = 0.4
        if select_action == 0:
            vy, vz = (-speed, -speed)
        elif select_action == 1:
            vy, vz = (0, -speed)
        elif select_action == 2:
            vy, vz = (speed, -speed)
        elif select_action == 3:
            vy, vz = (-speed, 0)
        elif select_action == 4:
            vy, vz = (0, 0)
        elif select_action == 5:
            vy, vz = (speed, 0)
        elif select_action == 6:
            vy, vz = (-speed, speed)
        elif select_action == 7:
            vy, vz = (0, speed)
        else:
            vy, vz = (speed, speed)

        # self.ctrl.update_target_rel((speed*0.5, vy, -vz))

        # # Execute action
        # self.drone.moveByVelocityBodyFrameAsync(speed, vy, vz, duration=1).join()
        #
        # # # Prevent swaying
        # self.drone.moveByVelocityAsync(vx=0, vy=0, vz=0, duration=1)

