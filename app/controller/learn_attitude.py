from app.models.attitude.learn_attitude_model import LearningAttitude


def learn_attitude(app_name: str, max_iter: int):
    if app_name == "attitude_discrete":
        env_name = "scripts:learn_attitude_ctrl_discrete_env-v0"
        policy = "MlpPolicy"
        max_integrate_time = 3
        random_start = False
    elif app_name == "attitude_continuous":
        env_name = "scripts:learn_attitude_ctrl_continuous_env-v0"
        policy = "MlpPolicy"
        max_integrate_time = 3
        random_start = False
    elif app_name == "attitude_fragment":
        env_name = "scripts:learn_attitude_ctrl_fragment_env-v0"
        policy = "MlpPolicy"
        max_integrate_time = 3
        random_start = False
    elif app_name == "attitude_test":
        env_name = "scripts:learn_attitude_ctrl_test_env-v0"
        policy = "MlpPolicy"
        max_integrate_time = 3
        random_start = False
    else:
        env_name = ""
        policy = "MlpPolicy"
        max_integrate_time = 3
        random_start = False

    learning_attitude = LearningAttitude(name=app_name, env_name=env_name, policy=policy,
                                         max_integrate_time=max_integrate_time,
                                         random_start=random_start)
    learning_attitude.learn(max_iter=max_iter)
