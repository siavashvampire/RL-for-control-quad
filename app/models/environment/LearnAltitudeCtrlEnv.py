import math
import os
import time

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
                   'prop_size': [6, 4.5], 'weight': 0.12}

CONTROLLER_PARAMETERS = {'Motor_limits': [2000, 5000],
                         'Tilt_limits': [-10, 10],
                         'Yaw_Control_Limits': [-3000, 3000],
                         'Z_XY_offset': 500,
                         'Linear_PID': {'P': [300, 300, 7000], 'I': [0.04, 0.04, 4.5], 'D': [450, 450, 5000]},
                         'Linear_To_Angular_Scaler': [1, 1, 0],
                         'Yaw_Rate_Scaler': 1,
                         'Angular_PID': {'P': [22000, 22000, 1500], 'I': [0, 0, 1.2], 'D': [12000, 12000, 0]},
                         }

PI = math.pi


class LearnAltitudeCtrlMain(gym.Env):
    random_start: bool
    settling_flag: bool = False
    settling_time: int = 20
    perv_action: int = 0
    max_integrate_time: int
    count: int
    plot_count: int
    reward_list: list[int] = []
    obs1_list: list[float] = []
    obs2_list: list[float] = []
    obs3_list: list[float] = []
    err1_list: list[float] = []
    err2_list: list[float] = []
    time_list: list[float] = []
    name: str = "altitude_main"
    fp: list = [[100, 1000],
                [0.01, 0.1],
                [100, 1000],
                [100, 1000],
                [0.01, 0.1],
                [100, 1000],
                [5000, 15000],
                [3, 10],
                [2000, 10000]]

    # {'P': [300, 300, 7000], 'I': [0.04, 0.04, 4.5], 'D': [450, 450, 5000]}

    def __init__(self, count: int = 0, random_start: bool = True, max_integrate_time: int = 3):
        self.quad = Quadcopter(QUAD_PARAMETERS)

        self.ctrl = ControllerPID(get_state=self.quad.get_state,
                                  get_time=self.quad.get_time,
                                  actuate_motors=self.quad.set_motor_speeds,
                                  params=CONTROLLER_PARAMETERS)

        # self.quad.update = types.MethodType(self.quad_update, self.quad)
        # self.quad.state_dot = types.MethodType(self.quad_state_dot, self.quad)

        self.observation_space = gym.spaces.Box(low=-50, high=50, shape=(1, 15), dtype=np.float32)
        self.action_space = gym.spaces.MultiDiscrete([3, 3, 3,
                                                      3, 3, 3,
                                                      3, 3, 3])

        self.info = {"collision": False}
        self.count = count

        self.random_start = random_start
        self.max_integrate_time = max_integrate_time

        self.plot_count = 6

    def step(self, action: np.ndarray):
        self.do_action(action)
        self.time_list.append(self.quad.integrate_time)
        obs, info = self.get_obs()
        # print(obs)
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
        self.reward_list = []
        self.obs1_list = []
        self.obs2_list = []
        self.obs3_list = []
        self.err1_list = []
        self.err2_list = []
        self.time_list = []
        self.count += 1
        obs, _ = self.get_obs()

        self.ctrl.start_thread()
        self.quad.start_thread()
        return obs

    def render(self):
        return self.get_obs()

    def setup_flight(self):
        self.settling_flag = False
        self.settling_time = 20
        self.quad.reset()
        self.ctrl.reset()

        # Get a random section
        if self.random_start:
            self.quad.set_position(
                (((-1, -1, -1) ** np.random.randint(0, 2, 3)) * (0.1 - np.random.random(3) * 0.05)).squeeze())
        else:
            self.quad.set_position((0, 0, 1))

        self.quad.set_orientation((0, 0, 0))
        # TODO:ma inja fght bara target sefr shoro kardim baad b fekr target tasadofi bashim

        self.ctrl.update_target((3, 3, 1.2))
        self.ctrl.update_yaw_target(0)

    def do_action(self, select_action: np.ndarray) -> None:
        # temp_pid = [[22000, 22000, 1500],
        #             [0, 0, 1.2],
        #             [12000, 12000, 0]]
        temp_pid = [[60000, 60000, 1500],
                    [0.1, 0.1, 1.2],
                    [12000, 12000, 0]]

        self.ctrl.set_ANGULAR_PID(temp_pid)

    def get_obs(self):
        self.info["collision"] = self.is_collision()
        obs, _ = self.ctrl.get_obs()
        k = self.ctrl.get_LINEAR_PID()

        for i in range(3):
            for j in range(3):
                if self.fp[3 * i + j][1] - self.fp[3 * i + j][0] == 0:
                    k[i][j] = 0
                else:
                    k[i][j] = np.divide(k[i][j], self.fp[3 * i + j][1] - self.fp[3 * i + j][0])

        k = k.reshape((1, 9))

        k = np.concatenate(([[obs[0][0], obs[1][0], obs[2][0], obs[0][1], obs[1][1], obs[2][1]]], k), axis=1)
        return k[0], self.info

    def compute_reward(self) -> (int, int):
        done: int = 0
        reward: int = 0
        error, error_dot = self.ctrl.get_diff_linear()

        error = abs(error) / 3
        error_dot = abs(error_dot) / 3

        self.err1_list.append(error)
        self.err2_list.append(error_dot)

        if error > 1 * 10:
            reward -= 50
        elif error > 0.75* 10:
            reward -= 30
        elif error > 0.5*10:
            reward -= 20
        elif error > 0.3*10:
            reward -= 10
        elif error > 0.2*10:
            reward -= 7
        elif error > 0.1*10:
            reward -= 1
        else:
            reward += 20
            if not self.settling_flag:
                self.settling_flag = True
                self.settling_time = self.quad.integrate_time

        if error_dot > 1*10:
            reward -= 50
        elif error_dot > 0.5*10:
            reward -= 30
        elif error_dot > 0.3*10:
            reward -= 10
        elif error_dot > 0.1*10:
            reward -= 5
        else:
            reward += 0

        if self.quad.integrate_time > self.max_integrate_time:
            done = 1

        if self.is_collision():
            reward = -200
            done = 1

        return reward, done

    def is_collision(self) -> bool:
        error = self.ctrl.get_error_linear()
        error = abs(error)
        # print(error)
        if abs(error[0]) > 0.75 * 10:
            return True
        if abs(error[1]) > 0.75 * 10:
            return True
        if abs(error[2]) > 2 * 10:
            return True
        return False

    def stop(self) -> None:
        self.quad.reset()
        self.ctrl.reset()

    def print_steps(self) -> None:
        figs: list[Figure] = []
        axes: list[Axes] = []

        titles = [
            'x(m) vs time(s)',
            'y(m) vs time(s)',
            'z(m) vs time(s)',
            'reward vs time(s)',
            'error(m) vs time(s)',
            'error dot(m/s) vs time(s)',
        ]

        x_label = [
            'time(s)',
            'time(s)',
            'time(s)',
            'time(s)',
            'time(s)',
            'time(s)',
        ]

        y_label = [
            'x(m)',
            'y(m)',
            'z(m)',
            'reward',
            'error(m)',
            'error dot(m/s)',
        ]

        for i in range(self.plot_count):
            figs.append(plt.figure())
            axes.append(figs[i].add_axes((0.15, 0.15, 0.7, 0.7)))
            axes[i].set_title(titles[i])
            axes[i].set_xlabel(x_label[i])
            axes[i].set_ylabel(y_label[i])

        axes[0].plot(self.time_list, self.obs1_list, linewidth=1)
        axes[1].plot(self.time_list, self.obs2_list, linewidth=1)
        axes[2].plot(self.time_list, self.obs3_list, linewidth=1)
        axes[3].plot(self.time_list, self.reward_list, linewidth=1)
        axes[4].plot(self.time_list, self.err1_list, linewidth=1)
        axes[5].plot(self.time_list, self.err2_list, linewidth=1)

        axes[0].scatter(self.time_list, self.obs1_list)
        axes[1].scatter(self.time_list, self.obs2_list)
        axes[2].scatter(self.time_list, self.obs3_list)
        axes[3].scatter(self.time_list, self.reward_list)
        axes[4].scatter(self.time_list, self.err1_list)
        axes[5].scatter(self.time_list, self.err2_list)

        result_dir = f'learn_result/{self.name}/{self.count}/'

        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        names = [
            '1.x vs time.png',
            '2.y vs time.png',
            '3.z vs time.png',
            '4.reward vs time.png',
            '5.error vs time.png',
            '6.error dot vs time.png',
        ]

        for i in range(self.plot_count):
            figs[i].savefig(result_dir + names[i])
            plt.close(figs[i])
            plt.close(figs[i])


class LearnAltitudeCtrlEnvContinuous(LearnAltitudeCtrlMain):
    def __init__(self, count: int = 0, random_start: bool = True, max_integrate_time: int = 3):
        super().__init__(count=count, random_start=random_start, max_integrate_time=max_integrate_time)
        self.name = "altitude_continuous"
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(9,), dtype=np.float32)

    def do_action(self, select_action: np.ndarray) -> None:
        super(LearnAltitudeCtrlEnvContinuous, self).do_action(select_action)
        select_action = select_action.reshape((3, 3))

        for i in range(3):
            for j in range(3):
                select_action[i][j] = select_action[i][j] * (self.fp[3 * i + j][1] - self.fp[3 * i + j][0]) + \
                                      self.fp[3 * i + j][0]

        select_action = np.array(select_action).transpose()
        self.ctrl.set_LINEAR_PID(select_action)


class LearnAltitudeCtrlEnvDiscrete(LearnAltitudeCtrlMain):
    def __init__(self, count: int = 0, random_start: bool = True, max_integrate_time: int = 3):
        super().__init__(count=count, random_start=random_start, max_integrate_time=max_integrate_time)
        self.name = "altitude_discrete"
        self.action_space = gym.spaces.MultiDiscrete([3, 3, 3,
                                                      3, 3, 3,
                                                      3, 3, 3])

    def do_action(self, select_action: np.ndarray) -> None:
        super(LearnAltitudeCtrlEnvDiscrete, self).do_action(select_action)
        select_action = select_action.reshape((3, 3))
        linear_PID = self.ctrl.get_LINEAR_PID()
        diff = np.array([[10, 0.0014 * linear_PID[0][0], 0.012 * linear_PID[0][0]],
                         [10, 0.0014 * linear_PID[1][0], 0.012 * linear_PID[1][0]],
                         [1000, 0.0014 * linear_PID[2][0], 0.012 * linear_PID[2][0]]])
        temp_pid = [[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]]

        for i in range(3):
            for j in range(3):
                if select_action[i][j] == 0:
                    temp_pid[i][j] += diff[i][j]  # increase p
                elif select_action[i][j] == 1:
                    temp_pid[i][j] -= diff[i][j]  # decrease p
                elif select_action[i][j] == 2:
                    temp_pid[i][j] += 0  # no action

                if temp_pid[i][j] < self.fp[i * 3 + j][0]:
                    temp_pid[i][j] = self.fp[i * 3 + j][0]
                if temp_pid[i][j] > self.fp[i * 3 + j][1]:
                    temp_pid[i][j] = self.fp[i * 3 + j][1]

        temp_pid = np.array(temp_pid).transpose()
        self.ctrl.set_LINEAR_PID(temp_pid)


class LearnAltitudeCtrlEnvFragment(LearnAltitudeCtrlMain):
    def __init__(self, count: int = 0, random_start: bool = True, max_integrate_time: int = 3):
        super().__init__(count=count, random_start=random_start, max_integrate_time=max_integrate_time)
        self.name = "altitude_fragment"

        self.action_space = gym.spaces.MultiDiscrete([15, 15, 15,
                                                      15, 15, 15,
                                                      15, 15, 15])

    def do_action(self, select_action: np.ndarray) -> None:
        super(LearnAltitudeCtrlEnvFragment, self).do_action(select_action)
        xp: list = [0, 14]

        temp_pid = np.zeros((1, 9))[0]

        for i in range(9):
            temp_pid[i] = np.interp(select_action[i], xp, self.fp[i])

        temp_pid = temp_pid.reshape((3, 3))
        temp_pid = np.array(temp_pid).transpose()

        self.ctrl.set_LINEAR_PID(temp_pid)


class LearnAltitudeCtrlEnvTest(LearnAltitudeCtrlMain):
    def __init__(self, count: int = 0, random_start: bool = True, max_integrate_time: int = 3):
        super().__init__(count=count, random_start=random_start, max_integrate_time=max_integrate_time)
        self.name = "altitude_test"

    def do_action(self, select_action: np.ndarray) -> None:
        super(LearnAltitudeCtrlEnvTest, self).do_action(select_action)
        # 'Linear_PID': {'P': [300, 300, 7000], 'I': [0.04, 0.04, 4.5], 'D': [450, 450, 5000]},
        temp_pid = [[300, 300, 7000],
                    [0.04, 0.04, 4.5],
                    [450, 450, 5000]]

        self.ctrl.set_LINEAR_PID(temp_pid)
