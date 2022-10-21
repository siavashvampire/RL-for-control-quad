import os
from typing import Union

import gym
import numpy as np
from selenium.webdriver.chrome.webdriver import WebDriver

from stable_baselines3 import PPO,A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from wandb.sdk.lib import RunDisabled
from wandb.sdk.wandb_run import Run

from app.RL_logging.RL_logging import open_log_web, close_tensorboard
from app.iteration_handler.iteration_handler import IterationHandler

import wandb
from wandb.integration.sb3 import WandbCallback


class LearningAttitude:
    run: Union[Run, RunDisabled, None]
    env: DummyVecEnv
    driver: WebDriver
    logdir: str
    models_dir: str
    iteration_handler: IterationHandler
    name: str
    env_name: str
    policy: str
    model: Union[PPO,A2C]
    max_integrate_time: int
    random_start: bool

    def __init__(self, name: str, env_name: str, policy: str, max_integrate_time: int = 15,
                 random_start: bool = True):
        self.name = name
        self.env_name = env_name
        self.policy = policy
        self.max_integrate_time = max_integrate_time
        self.random_start = random_start
        self.iteration_handler = IterationHandler(name)
        self.models_dir = f"models/{self.name}/"
        self.logdir = f"logs/{self.name}/"

    def learn(self, max_iter: int = np.inf):
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        self.iteration_handler.set_max_iter(max_iter)
        self.driver = open_log_web()

        time_steps = 10000


        # Create a DummyVecEnv
        self.env = DummyVecEnv([lambda: Monitor(
            gym.make(
                self.env_name,
                count=self.iteration_handler.read_count(),
                random_start=self.random_start,
                max_integrate_time=self.max_integrate_time
            )
        )])

        self.env.reset()

        iters = self.iteration_handler.read_iter()

        if iters == 0:
            if "A2C" in self.name:
                self.model = A2C(self.policy, self.env, verbose=1, device="cuda", tensorboard_log=self.logdir)
            else:
                self.model = PPO(self.policy, self.env, verbose=1, device="cuda", tensorboard_log=self.logdir)
        else:
            model_path = f"{self.models_dir}/{iters * time_steps}.zip"
            if "A2C" in self.name:
                self.model = A2C.load(model_path, env=self.env)
            else:
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

        # callback = WandbCallback(
        #     gradient_save_freq=500,
        #     model_save_path=f"{self.models_dir}/wandb/{self.run.id}",
        #     verbose=2,
        # )

        # callbacks.append(callback)
        kwargs = {"callback": callbacks}

        if "A2C" in self.name:
            tb_log_name = f"A2C"
        else:
            tb_log_name = f"PPO"

        print(f"run learning with name {self.name}")
        while self.iteration_handler.read_flag():
            iters += 1
            self.model.learn(total_timesteps=time_steps, reset_num_timesteps=False, tb_log_name=tb_log_name, **kwargs)
            self.model.save(f"{self.models_dir}/{time_steps * iters}")
            self.iteration_handler.write_iter(iters)
            self.iteration_handler.write_count(self.env.envs[0].count)

        self.env.envs[0].stop()
        self.driver.close()
        close_tensorboard()
        # self.run.finish()
