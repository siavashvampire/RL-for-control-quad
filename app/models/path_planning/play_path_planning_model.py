import jdatetime
import os

import gym
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from tqdm import tqdm

from app.iteration_handler.iteration_handler import IterationHandler
from scripts.airsim_env import open_airsim_test_env


class PlayPathPlanning:
    continue_flag: bool
    num_fig: int
    model: PPO
    times: int
    axes: list[Axes]
    figs: list[Figure]
    models_dir: str
    iteration_handler: IterationHandler
    env_name: str
    file_name: str
    name: str
    env: VecTransposeImage
    random_start: bool

    def __init__(self, name: str, file_name: str, env_name: str, times: int = 5, continue_flag: bool = False,
                 random_start: bool = True):
        self.name = name
        self.file_name = file_name
        self.env_name = env_name
        self.iteration_handler = IterationHandler(name)
        self.random_start = random_start
        self.continue_flag = continue_flag

        self.times = times

        self.models_dir = f"models/{name}/"

        self.num_fig = 4
        self.figs = []
        self.axes = []
        self.init_fig()

        np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

    def init_fig(self) -> None:

        for i in range(self.num_fig):
            self.figs.append(plt.figure())
            self.axes.append(self.figs[i].add_axes((0.1, 0.1, 0.8, 0.8)))

        titles = ['x vs time',
                  'y vs time',
                  'z vs time',
                  'reward vs time']

        xlabel = ['time',
                  'time',
                  'time',
                  'time', ]

        ylabel = ['x',
                  'y',
                  'z',
                  'reward']

        for i in range(self.num_fig):
            self.axes[i].set_title(titles[i])
            self.axes[i].set_xlabel(xlabel[i])
            self.axes[i].set_ylabel(ylabel[i])

    def play(self) -> None:
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

        with open('scripts/config.yml', 'r') as f:
            env_config = yaml.safe_load(f)

        # Create a DummyVecEnv
        self.env = VecTransposeImage(DummyVecEnv([lambda: Monitor(
            gym.make(
                "scripts:learn_path_env-v0",
                ip_address="127.0.0.1",
                image_shape=(50, 50, 3),
                env_config=env_config["TrainEnv"],
                continue_flag=self.continue_flag,
                random_start=self.random_start
            )
        )]))

        open_airsim_test_env()

        self.model = PPO.load(env=self.env, path=self.models_dir + self.file_name)

        for i in tqdm(range(self.times), colour='green'):
            obs = self.env.reset()
            done = False

            x_temp = []
            y_temp = []
            z_temp = []
            y_ref_temp = []
            z_ref_temp = []
            reward_temp = []

            x, y, z = self.env.envs[0].drone.simGetVehiclePose().position
            y_ref, z_ref = self.env.envs[0].target_pos

            x_temp.append(x)
            y_temp.append(y)
            z_temp.append(z)
            y_ref_temp.append(y_ref)
            z_ref_temp.append(z_ref)
            reward_temp.append(0)

            while not done:
                action, _state = self.model.predict(obs)
                obs, rewards, done, info = self.env.step(action)

                if not done:
                    x, y, z = self.env.envs[0].drone.simGetVehiclePose().position
                    y_ref, z_ref = self.env.envs[0].target_pos

                    x_temp.append(x)
                    y_temp.append(y)
                    z_temp.append(z)
                    y_ref_temp.append(y_ref)
                    z_ref_temp.append(z_ref)
                    reward_temp.append(rewards)
                else:
                    reward_temp.append(rewards)
            self.axes[0].plot(x_temp, linewidth=1)
            self.axes[1].plot(y_temp, linewidth=1)
            self.axes[1].plot(y_ref_temp, linewidth=1)
            self.axes[2].plot(z_temp, linewidth=1)
            self.axes[2].plot(z_ref_temp, linewidth=1)
            self.axes[3].plot(reward_temp, linewidth=1)
            # self.axes[1].plot(x_temp, np.array(phi_temp) * 180 / math.pi, linewidth=1)
            # self.axes[2].plot(x_temp, np.array(gamma_temp) * 180 / math.pi, linewidth=1)
            # self.axes[3].plot(x_temp, theta_dot_temp, linewidth=1)
            # self.axes[4].plot(x_temp, phi_dot_temp, linewidth=1)
            # self.axes[5].plot(x_temp, gamma_dot_temp, linewidth=1)
            # trust_temp = np.array(trust_temp)
            # self.axes[7].plot(x_temp, trust_temp[:, 0], linewidth=1)
            # self.axes[7].plot(x_temp, trust_temp[:, 1], linewidth=1)
            # self.axes[7].plot(x_temp, trust_temp[:, 2], linewidth=1)
            # self.axes[7].plot(x_temp, trust_temp[:, 3], linewidth=1)
            # self.axes[8].plot(x_temp, kp_temp, linewidth=1)
            # x_temp.append(x_temp[-1] + 0.02)
            # self.axes[6].plot(x_temp, reward_temp, linewidth=1)

        self.env.envs[0].stop()

        self.print_plot(file_name=jdatetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))

    def print_plot(self, file_name: str):
        for i in range(self.num_fig):
            self.axes[i].legend(range(1, self.times + 1))

        # self.axes[0].set_ybound(-0.8, 0.8)
        # self.axes[1].set_ybound(-0.8, 0.8)
        # self.axes[2].set_ybound(-0.8, 0.8)
        # self.axes[3].set_ybound(-0.5, 0.5)
        # self.axes[4].set_ybound(-0.5, 0.5)
        # self.axes[5].set_ybound(-0.5, 0.5)
        # self.axes[6].set_ybound(-100, 100)

        # angular_p = str(env.envs[0].ctrl.get_angular_p_theta())
        result_dir = f'Result/{self.name}/{file_name}/'

        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        names = ['1.x vs time.png',
                 '2.y vs time.png',
                 '3.z vs time.png',
                 '4.reward vs time.png']

        for i in range(self.num_fig):
            self.figs[i].savefig(result_dir + names[i])
            plt.close(self.figs[i])
