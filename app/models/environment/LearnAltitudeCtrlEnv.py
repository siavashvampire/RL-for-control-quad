import math
import os
import time
import types

import gym
import numpy as np
from app.models.controller.controller import ControllerPID
from app.models.quadcopter_dynamics.quadcopter import Quadcopter

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

TIME_SCALING = 1.0  # Any positive number(Smaller is faster). 1.0->Real Time, 0.0->Run as fast as possible
QUAD_DYNAMICS_UPDATE = 0.002  # seconds
CONTROLLER_DYNAMICS_UPDATE = 0.005  # seconds

QUAD_PARAMETERS = {'Motor_limits': [2000, 5000], 'position': [0, 0, 1], 'orientation': [0, 0, 0], 'L': 0.3, 'r': 0.1,
                   'prop_size': [6, 4.5], 'weight': 1.2}

CONTROLLER_PARAMETERS = {'Motor_limits': [2000, 5000],
                         'Tilt_limits': [-10, 10],
                         'Yaw_Control_Limits': [-3000, 3000],
                         'Z_XY_offset': 500,
                         'Linear_PID': {'P': [300, 300, 7000], 'I': [0.04, 0.04, 4.5], 'D': [450, 450, 5000]},
                         'Linear_To_Angular_Scaler': [1, 1, 0],
                         'Yaw_Rate_Scaler': 1,
                         'Angular_PID': {'P': [0, 0, 60000], 'I': [0, 0, 1.2], 'D': [12000, 12000, 10000]},
                         }

PI = math.pi


class LearnAltitudeCtrlMain(gym.Env):
    random_start: bool
    settling_flag: bool = False
    settling_time: int = 20
    perv_action: int = 0
    max_integrate_time: int
    last_p: int = 0
    count: int
    action_list: list[float] = []
    p_list: list[float] = []
    reward_list: list[int] = []
    obs1_list: list[float] = []
    obs2_list: list[float] = []
    obs3_list: list[float] = []
    time_list: list[float] = []
    name: str = "altitude_main"

    def __init__(self, count: int = 0, random_start: bool = True, max_integrate_time: int = 3):
        self.quad = Quadcopter(QUAD_PARAMETERS)

        self.ctrl = ControllerPID(get_state=self.quad.get_state,
                                  get_time=self.quad.get_time,
                                  actuate_motors=self.quad.set_motor_speeds,
                                  params=CONTROLLER_PARAMETERS)

        self.quad.update = types.MethodType(self.quad_update, self.quad)
        self.quad.state_dot = types.MethodType(self.quad_state_dot, self.quad)

        self.observation_space = gym.spaces.Box(low=-50, high=50, shape=(1, 3), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)

        self.info = {"collision": False}
        self.count = count

        self.random_start = random_start
        self.max_integrate_time = max_integrate_time

    def step(self, action):
        self.do_action(action)
        self.action_list.append(action)
        self.p_list.append(self.last_p)
        self.time_list.append(self.quad.integrate_time)
        obs, info = self.get_obs()
        reward, done = self.compute_reward()
        self.reward_list.append(reward)
        self.obs1_list.append(obs[0])
        self.obs2_list.append(obs[1])
        self.obs3_list.append(obs[2])
        time.sleep(0.1)
        return obs, reward, done, info

    def reset(self):
        self.setup_flight()
        self.print_steps()
        self.action_list = []
        self.p_list = []
        self.reward_list = []
        self.obs1_list = []
        self.obs2_list = []
        self.obs3_list = []
        self.time_list = []
        self.count += 1
        obs, _ = self.get_obs()
        return obs

    def render(self):
        return self.get_obs()

    def setup_flight(self):
        self.settling_flag = False
        self.settling_time = 20
        # self.last_p = 0
        self.quad.reset()
        self.ctrl.reset()

        # Get a random section
        if self.random_start:
            self.quad.set_orientation(
                (((-1, -1, -1) ** np.random.randint(0, 2, 3)) * (0.1 - np.random.random(3) * 0.05)).squeeze())
        else:
            self.quad.set_orientation((0.5, 0.5, 0))

        obs, _ = self.get_obs()

        self.quad.set_position((0, 0, 1))
        # TODO:ma inja fght bara target sefr shoro kardim baad b fekr target tasadofi bashim

        self.ctrl.update_target((0, 0, 1))
        self.ctrl.update_yaw_target(0)

        self.ctrl.start_thread()
        self.quad.start_thread()

    def do_action(self, select_action):
        pass

    def get_obs(self):
        self.info["collision"] = self.is_collision()
        _, obs = self.ctrl.get_obs()
        k_p = self.ctrl.get_angular_p_theta() / 40000

        return [obs[0][0], obs[0][1], k_p], self.info

    def compute_reward(self):
        done = 0
        reward = 0
        e_theta, theta_dot = self.ctrl.get_diff_angular_theta()
        e_theta = abs(e_theta * 180 / PI)
        theta_dot = abs(theta_dot)

        if e_theta > 25:
            reward -= 50
        elif e_theta > 20:
            reward -= 30
        elif e_theta > 15:
            reward -= 20
        elif e_theta > 10:
            reward -= 10
        elif e_theta > 5:
            reward -= 7
        elif e_theta > 2:
            reward -= 1
        else:
            reward += 20
            if not self.settling_flag:
                self.settling_flag = True
                self.settling_time = self.quad.integrate_time

        if theta_dot > 1:
            reward -= 50
        elif theta_dot > 0.5:
            reward -= 30
        elif theta_dot > 0.3:
            reward -= 10
        elif theta_dot > 0.1:
            reward -= 5
        else:
            reward += 0

        # reward = self.ctrl.get_diff_angular_theta() * -100
        # if self.settling_flag:
        #     reward -= (self.settling_time - 2) * 100
        # elif self.quad.integrate_time > 2:
        #     reward -= (self.quad.integrate_time - 2) * 100

        if self.quad.integrate_time > self.max_integrate_time:
            done = 1

        if self.is_collision():
            reward = -200
            done = 1

        # print("reward = ", reward)
        # print("settling_time = ", self.settling_time)

        return reward, done

    def is_collision(self) -> bool:
        if abs(self.quad.state[6]) > 3.14 * 0.25:
            return True
        if abs(self.quad.state[7]) > 3.14 * 0.25:
            return True
        if abs(self.quad.state[8]) > 3.14 * 0.25:
            return True
        return False

    @staticmethod
    def quad_update(quad: Quadcopter, dt):
        quad.ode.set_initial_value(quad.state, 0)

        quad.state = quad.ode.integrate(quad.ode.t + dt)
        quad.state[0:2] = np.zeros(2)
        quad.state[3:6] = np.zeros(3)
        quad.state[2] = 1
        quad.state[6:9] = quad.wrap_angle(quad.state[6:9])
        quad.integrate_time += dt

    @staticmethod
    def quad_state_dot(quad: Quadcopter, time, state):
        state_dot = np.zeros(12)
        state_dot[6] = quad.state[9]
        state_dot[7] = quad.state[10]
        state_dot[8] = quad.state[11]
        # The angular accelerations
        omega = quad.state[9:12]
        tau = np.array([quad.L * (quad.m1.thrust - quad.m3.thrust),
                        quad.L * (quad.m2.thrust - quad.m4.thrust),
                        quad.b * (quad.m1.thrust - quad.m2.thrust + quad.m3.thrust - quad.m4.thrust)])
        omega_dot = np.dot(quad.invI, (tau - np.cross(omega, np.dot(quad.I, omega))))
        state_dot[9] = omega_dot[0]
        state_dot[10] = omega_dot[1]
        state_dot[11] = omega_dot[2]
        return state_dot

    def stop(self):
        self.quad.reset()
        self.ctrl.reset()

    def print_steps(self):
        figs: list[Figure] = []
        axes: list[Axes] = []

        for i in range(5):
            figs.append(plt.figure())
            axes.append(figs[i].add_axes((0.1, 0.1, 0.8, 0.8)))

        axes[0].set_title('theta vs time')
        axes[0].set_xlabel('time')
        axes[0].set_ylabel('theta')

        axes[1].set_title('action vs time')
        axes[1].set_xlabel('time')
        axes[1].set_ylabel('action')

        axes[2].set_title('reward vs time')
        axes[2].set_xlabel('time')
        axes[2].set_ylabel('reward')

        axes[3].set_title('theta dot vs time')
        axes[3].set_xlabel('time')
        axes[3].set_ylabel('theta dot')

        axes[4].set_title('p vs time')
        axes[4].set_xlabel('time')
        axes[4].set_ylabel('P')

        axes[0].plot(self.time_list, self.obs1_list, linewidth=1)
        axes[1].plot(self.time_list, self.action_list, linewidth=1)
        axes[2].plot(self.time_list, self.reward_list, linewidth=1)
        axes[3].plot(self.time_list, self.obs2_list, linewidth=1)
        axes[4].plot(self.time_list, self.p_list, linewidth=1)

        axes[0].scatter(self.time_list, self.obs1_list)
        axes[1].scatter(self.time_list, self.action_list)
        axes[2].scatter(self.time_list, self.reward_list)
        axes[3].scatter(self.time_list, self.obs2_list)
        axes[4].scatter(self.time_list, self.p_list)

        result_dir = f'learn_result/{self.name}/{self.count}/'

        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        figs[0].savefig(result_dir + '1.theta vs time.png')
        figs[1].savefig(result_dir + '2.action vs time.png')
        figs[2].savefig(result_dir + '3.reward vs time.png')
        figs[3].savefig(result_dir + '4.theta dot vs time.png')
        figs[4].savefig(result_dir + '5.p vs time.png')

        for i in range(5):
            plt.close(figs[i])


class LearnAltitudeCtrlEnvContinuous(LearnAltitudeCtrlMain):
    def __init__(self, count: int = 0, random_start: bool = True, max_integrate_time: int = 3):
        super().__init__(count=count, random_start=random_start, max_integrate_time=max_integrate_time)
        self.name = "altitude_continuous"
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

    def do_action(self, select_action):
        angular_p_theta = select_action[0] * 40000
        self.ctrl.set_angular_p_theta(angular_p_theta)
        self.ctrl.set_angular_p_phi(angular_p_theta)
        self.last_p = angular_p_theta

    def step(self, action):
        self.do_action(action)
        self.action_list.append(action[0])
        self.p_list.append(self.last_p)
        self.time_list.append(self.quad.integrate_time)
        obs, info = self.get_obs()
        reward, done = self.compute_reward()
        self.reward_list.append(reward)
        self.obs1_list.append(obs[0])
        self.obs2_list.append(obs[1])
        self.obs3_list.append(obs[2])
        time.sleep(0.1)
        return obs, reward, done, info


class LearnAltitudeCtrlEnvDiscrete(LearnAltitudeCtrlMain):
    def __init__(self, count: int = 0, random_start: bool = True, max_integrate_time: int = 3):
        super().__init__(count=count, random_start=random_start, max_integrate_time=max_integrate_time)
        self.name = "altitude_discrete"
        self.action_space = gym.spaces.Discrete(3)

    def do_action(self, select_action):
        angular_p_theta = self.ctrl.get_angular_p_theta()
        diff = 5000
        if select_action == 0:
            angular_p_theta += diff  # increase p
        elif select_action == 1:
            angular_p_theta -= diff  # decrease p
        elif select_action == 2:
            angular_p_theta += 0  # no action

        if angular_p_theta < 0:
            angular_p_theta = 0

        self.ctrl.set_angular_p_theta(angular_p_theta)
        self.ctrl.set_angular_p_phi(angular_p_theta)
        self.last_p = angular_p_theta


class LearnAltitudeCtrlEnvFragment(LearnAltitudeCtrlMain):
    def __init__(self, count: int = 0, random_start: bool = True, max_integrate_time: int = 3):
        super().__init__(count=count, random_start=random_start, max_integrate_time=max_integrate_time)
        self.name = "altitude_fragment"
        self.action_space = gym.spaces.Discrete(15)

    def do_action(self, select_action):
        angular_p_theta = 0

        if select_action == 0:
            angular_p_theta = 5
        elif select_action == 1:
            angular_p_theta = 10
        elif select_action == 2:
            angular_p_theta = 15
        elif select_action == 3:
            angular_p_theta = 20
        elif select_action == 4:
            angular_p_theta = 25
        elif select_action == 5:
            angular_p_theta = 30
        elif select_action == 6:
            angular_p_theta = 35
        elif select_action == 7:
            angular_p_theta = 40
        elif select_action == 8:
            angular_p_theta = 45
        elif select_action == 9:
            angular_p_theta = 50
        elif select_action == 10:
            angular_p_theta = 55
        elif select_action == 11:
            angular_p_theta = 60
        elif select_action == 12:
            angular_p_theta = 0
        elif select_action == 13:
            angular_p_theta = 65
        elif select_action == 14:
            angular_p_theta = 70

        angular_p_theta *= 1000
        self.ctrl.set_angular_p_theta(angular_p_theta)
        self.ctrl.set_angular_p_phi(angular_p_theta)
        self.last_p = angular_p_theta


class LearnAltitudeCtrlEnvTest(LearnAltitudeCtrlMain):
    def __init__(self, count: int = 0, random_start: bool = True, max_integrate_time: int = 3):
        super().__init__(count=count, random_start=random_start, max_integrate_time=max_integrate_time)
        self.name = "altitude_test"

    def do_action(self, select_action):
        angular_p_theta = 22000
        self.ctrl.set_angular_p_theta(angular_p_theta)
        self.ctrl.set_angular_p_phi(angular_p_theta)
        self.last_p = angular_p_theta
