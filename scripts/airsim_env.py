import subprocess


def open_airsim_train_env():
    subprocess.Popen('cmd /k "cd TrainEnv && Trainenv.exe"', shell=True)


def open_airsim_test_env():
    subprocess.Popen('cmd /k "cd TestEnv && start TestEnv.exe"', shell=True)
