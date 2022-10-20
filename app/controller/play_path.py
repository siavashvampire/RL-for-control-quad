from app.models.path_planning.play_path_planning_model import PlayPathPlanning


def play_path(app_name: str):
    if app_name == "path_planning":
        env_name = "scripts:learn_path_env-v0"
        file_name = "best_model.zip"
        times = 2
        random_start = True
    else:
        env_name = ""
        file_name = ""
        times = 5
        random_start = True

    learning_path = PlayPathPlanning(name=app_name, file_name=file_name, env_name=env_name,
                                     times=times, random_start=random_start)
    learning_path.play()
