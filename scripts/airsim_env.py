import subprocess


def open_airsim_train_env():
    subprocess.Popen('cmd /k "start TrainEnv/Trainenv.exe"', shell=True)


def open_airsim_test_env():
    subprocess.Popen('cmd /k "start TestEnv/TestEnv.exe"', shell=True)
