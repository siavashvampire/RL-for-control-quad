import os
from typing import Union

import gym
import yaml
import numpy as np
from selenium.webdriver.chrome.webdriver import WebDriver

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import EvalCallback
from wandb.sdk.lib import RunDisabled
from wandb.sdk.wandb_run import Run
from scripts.airsim_env import open_airsim_train_env

from app.RL_logging.RL_logging import open_log_web
from app.iteration_handler.iteration_handler import IterationHandler


class LearningPathPlanning:
    run: Union[Run, RunDisabled, None]
    env: VecTransposeImage
    driver: WebDriver
    logdir: str
    models_dir: str
    iteration_handler: IterationHandler
    name: str
    env_name: str
    policy: str
    model: PPO
    max_integrate_time: int
    random_start: bool
    continue_flag: bool

    def __init__(self, name: str, env_name: str, policy: str,continue_flag:bool = False, random_start: bool = True):
        self.name = name
        self.env_name = env_name
        self.policy = policy
        self.random_start = random_start
        self.continue_flag = continue_flag
        self.iteration_handler = IterationHandler(name)
        self.models_dir = f"models/{self.name}/"
        self.logdir = f"logs/{self.name}/"

    def learn(self, max_iter: int = np.inf):
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)

        self.iteration_handler.set_max_iter(max_iter)
        open_airsim_train_env()
        self.driver = open_log_web()

        with open('./scripts/config.yml', 'r') as f:
            env_config = yaml.safe_load(f)

        time_steps = 10000

        # Create a DummyVecEnv
        env = DummyVecEnv([lambda: Monitor(
            gym.make(
                "scripts:learn_path_env-v0",
                ip_address="127.0.0.1",
                image_shape=(50, 50, 3),
                env_config=env_config["TrainEnv"],
                random_start = self.random_start,
                continue_flag = self.continue_flag

            )
        )])

        self.env = VecTransposeImage(env)
        self.env.reset()

        iters = self.iteration_handler.read_iter()

        if iters == 0:
            self.model = PPO(self.policy, env, verbose=1, seed=42, device="cuda", tensorboard_log=self.logdir)
        else:
            model_path = f"{self.models_dir}/{iters * time_steps}.zip"
            self.model = PPO.load(model_path, env=self.env)

        callbacks = []
        eval_callback = EvalCallback(
            self.env,
            callback_on_new_best=None,
            n_eval_episodes=4,
            best_model_save_path=self.models_dir,
            log_path="../../..",
            eval_freq=500,
        )

        callbacks.append(eval_callback)
        kwargs = {"callback": callbacks}

        print(f"run learning with name {self.name}")
        while self.iteration_handler.read_flag():
            iters += 1
            self.model.learn(total_timesteps=time_steps, reset_num_timesteps=False, tb_log_name=f"PPO", **kwargs)
            self.model.save(f"{self.models_dir}/{time_steps * iters}")
            self.iteration_handler.write_iter(iters)

        self.env.envs[0].stop()
        self.driver.close()
