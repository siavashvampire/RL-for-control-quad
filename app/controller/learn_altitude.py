from app.models.altitude.learn_altitude_model import LearningAltitude


def learn_altitude(app_name: str, max_iter: int):
    if app_name == "altitude_discrete":
        env_name = "scripts:learn_altitude_ctrl_discrete_env-v0"
        policy = "MlpPolicy"
        max_integrate_time = 3
        random_start = False
    elif app_name == "altitude_continuous":
        env_name = "scripts:learn_altitude_ctrl_continuous_env-v0"
        policy = "MlpPolicy"
        max_integrate_time = 3
        random_start = False
    elif app_name == "altitude_fragment":
        env_name = "scripts:learn_altitude_ctrl_fragment_env-v0"
        policy = "MlpPolicy"
        max_integrate_time = 3
        random_start = False
    elif app_name == "altitude_test":
        env_name = "scripts:learn_altitude_ctrl_test_env-v0"
        policy = "MlpPolicy"
        max_integrate_time = 3
        random_start = False
    else:
        env_name = ""
        policy = "MlpPolicy"
        max_integrate_time = 3
        random_start = False

    learning_altitude = LearningAltitude(name=app_name, env_name=env_name, policy=policy,
                                         max_integrate_time=max_integrate_time, random_start=random_start)
    learning_altitude.learn(max_iter=max_iter)
