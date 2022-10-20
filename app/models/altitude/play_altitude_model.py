import math
import os

import gym
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import tqdm

from app.iteration_handler.iteration_handler import IterationHandler


class PlayAltitude:
    model: PPO
    times: int
    axes: list[Axes]
    figs: list[Figure]
    max_integrate_time: int
    models_dir: str
    iteration_handler: IterationHandler
    env_name: str
    file_name: str
    name: str
    env: DummyVecEnv
    random_start: bool

    def __init__(self, name: str, file_name: str, env_name: str, max_integrate_time: int = 15, times: int = 5,
                 random_start: bool = True):
        self.name = name
        self.file_name = file_name
        self.env_name = env_name
        self.iteration_handler = IterationHandler(name)
        self.max_integrate_time = max_integrate_time
        self.random_start = random_start

        self.times = times

        self.models_dir = f"models/{name}/"

        self.figs = []
        self.axes = []
        self.init_fig()

        np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

    def init_fig(self) -> None:
        for i in range(9):
            self.figs.append(plt.figure())
            self.axes.append(self.figs[i].add_axes((0.1, 0.1, 0.8, 0.8)))

        self.axes[0].set_title('theta vs time')
        self.axes[0].set_xlabel('time')
        self.axes[0].set_ylabel('theta')

        self.axes[1].set_title('phi vs time')
        self.axes[1].set_xlabel('time')
        self.axes[1].set_ylabel('phi')

        self.axes[2].set_title('gamma vs time')
        self.axes[2].set_xlabel('time')
        self.axes[2].set_ylabel('gamma')

        self.axes[3].set_title('theta dot vs time')
        self.axes[3].set_xlabel('time')
        self.axes[3].set_ylabel('theta dot')

        self.axes[4].set_title('phi dot vs time')
        self.axes[4].set_xlabel('time')
        self.axes[4].set_ylabel('phi dot')

        self.axes[5].set_title('gamma dot vs time')
        self.axes[5].set_xlabel('time')
        self.axes[5].set_ylabel('gamma dot')

        self.axes[6].set_title('reward vs time')
        self.axes[6].set_xlabel('time')
        self.axes[6].set_ylabel('reward')

        self.axes[7].set_title('trust vs time')
        self.axes[7].set_xlabel('time')
        self.axes[7].set_ylabel('trust')

        self.axes[8].set_title('P vs time')
        self.axes[8].set_xlabel('time')
        self.axes[8].set_ylabel('P')

    def play(self) -> None:
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

        # Create a DummyVecEnv
        self.env = DummyVecEnv([lambda: Monitor(
            gym.make(
                self.env_name,
                count=self.iteration_handler.read_count(),
                random_start=self.random_start,
                max_integrate_time=self.max_integrate_time
            )
        )])

        self.model = PPO.load(env=self.env, path=self.models_dir + self.file_name)

        for i in tqdm(range(self.times), colour='green'):
            obs = self.env.reset()
            done = False
            time_temp = []
            theta_temp = []
            theta_dot_temp = []
            phi_temp = []
            phi_dot_temp = []
            gamma_temp = []
            gamma_dot_temp = []
            reward_temp = []
            trust_temp = []
            kp_temp = []

            orientation = self.env.envs[0].quad.get_orientation()
            angular_rate = self.env.envs[0].quad.get_angular_rate()
            time_temp.append(self.env.envs[0].quad.integrate_time)

            theta_temp.append(orientation[0])
            phi_temp.append(orientation[1])
            gamma_temp.append(orientation[2])
            theta_dot_temp.append(angular_rate[0])
            phi_dot_temp.append(angular_rate[1])
            gamma_dot_temp.append(angular_rate[2])
            reward_temp.append(0)
            trust_temp.append([0, 0, 0, 0])
            kp_temp.append(0)

            while not done:
                action, _state = self.model.predict(obs)
                obs, rewards, done, info = self.env.step(action)

                if not done:
                    orientation = self.env.envs[0].quad.get_orientation()
                    angular_rate = self.env.envs[0].quad.get_angular_rate()
                    time_temp.append(self.env.envs[0].quad.integrate_time)

                    theta_temp.append(orientation[0])
                    phi_temp.append(orientation[1])
                    gamma_temp.append(orientation[2])
                    theta_dot_temp.append(angular_rate[0])
                    phi_dot_temp.append(angular_rate[1])
                    gamma_dot_temp.append(angular_rate[2])
                    reward_temp.append(rewards)
                    trust_temp.append(self.env.envs[0].quad.get_trust())
                    kp_temp.append(self.env.envs[0].last_p)
                else:
                    reward_temp.append(rewards)

            self.axes[0].plot(time_temp, np.array(theta_temp)*180/math.pi, linewidth=1)
            self.axes[1].plot(time_temp, np.array(phi_temp)*180/math.pi, linewidth=1)
            self.axes[2].plot(time_temp, np.array(gamma_temp)*180/math.pi, linewidth=1)
            self.axes[3].plot(time_temp, theta_dot_temp, linewidth=1)
            self.axes[4].plot(time_temp, phi_dot_temp, linewidth=1)
            self.axes[5].plot(time_temp, gamma_dot_temp, linewidth=1)
            trust_temp = np.array(trust_temp)
            self.axes[7].plot(time_temp, trust_temp[:, 0], linewidth=1)
            self.axes[7].plot(time_temp, trust_temp[:, 1], linewidth=1)
            self.axes[7].plot(time_temp, trust_temp[:, 2], linewidth=1)
            self.axes[7].plot(time_temp, trust_temp[:, 3], linewidth=1)
            self.axes[8].plot(time_temp, kp_temp, linewidth=1)
            time_temp.append(time_temp[-1] + 0.02)
            self.axes[6].plot(time_temp, reward_temp, linewidth=1)

        self.env.envs[0].stop()

        self.print_plot(file_name=str(self.env.envs[0].last_p))

    def print_plot(self, file_name: str):
        self.axes[0].legend(range(1, self.times + 1))
        self.axes[1].legend(range(1, self.times + 1))
        self.axes[2].legend(range(1, self.times + 1))
        self.axes[3].legend(range(1, self.times + 1))
        self.axes[4].legend(range(1, self.times + 1))
        self.axes[5].legend(range(1, self.times + 1))
        self.axes[6].legend(range(1, self.times + 1))
        self.axes[7].legend(range(1, self.times + 1))
        self.axes[8].legend(range(1, self.times + 1))

        # self.axes[0].set_ybound(-0.8, 0.8)
        # self.axes[1].set_ybound(-0.8, 0.8)
        # self.axes[2].set_ybound(-0.8, 0.8)
        # self.axes[3].set_ybound(-0.5, 0.5)
        # self.axes[4].set_ybound(-0.5, 0.5)
        # self.axes[5].set_ybound(-0.5, 0.5)
        self.axes[6].set_ybound(-100, 100)

        # angular_p = str(env.envs[0].ctrl.get_angular_p_theta())
        result_dir = f'Result/{self.name}/{file_name}/'

        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        self.figs[0].savefig(result_dir + '1.theta vs time.png')
        self.figs[1].savefig(result_dir + '2.phi vs time.png')
        self.figs[2].savefig(result_dir + '3.gamma vs time.png')
        self.figs[3].savefig(result_dir + '4.theta dot vs time.png')
        self.figs[4].savefig(result_dir + '5.phi dot vs time.png')
        self.figs[5].savefig(result_dir + '6.gamma dot vs time.png')
        self.figs[6].savefig(result_dir + '7.reward vs time.png')
        self.figs[7].savefig(result_dir + '8.trust vs time.png')
        self.figs[8].savefig(result_dir + '9.P vs time.png')

        for i in range(9):
            plt.close(self.figs[i])