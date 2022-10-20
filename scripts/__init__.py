from app.models.environment.LearnAttitudeCtrlEnv import LearnAttitudeCtrlEnvDiscrete, LearnAttitudeCtrlEnvContinuous, \
    LearnAttitudeCtrlEnvFragment, LearnAttitudeCtrlEnvTest

from app.models.environment.LearnAltitudeCtrlEnv import LearnAltitudeCtrlEnvDiscrete, LearnAltitudeCtrlEnvContinuous, \
    LearnAltitudeCtrlEnvFragment, LearnAltitudeCtrlEnvTest

from app.models.environment.LearnPathEnv import LearnPathEnv

from gym.envs.registration import register

# Register AirSim environment as a gym environment
register(
    id="learn_attitude_ctrl_continuous_env-v0", entry_point="scripts:LearnAttitudeCtrlEnvContinuous",
)

# Register AirSim environment as a gym environment
register(
    id="learn_attitude_ctrl_discrete_env-v0", entry_point="scripts:LearnAttitudeCtrlEnvDiscrete",
)

# Register AirSim environment as a gym environment
register(
    id="learn_attitude_ctrl_fragment_env-v0", entry_point="scripts:LearnAttitudeCtrlEnvFragment",
)

# Register AirSim environment as a gym environment
register(
    id="learn_attitude_ctrl_test_env-v0", entry_point="scripts:LearnAttitudeCtrlEnvTest",
)

# Register AirSim environment as a gym environment
register(
    id="learn_altitude_ctrl_continuous_env-v0", entry_point="scripts:LearnAltitudeCtrlEnvContinuous",
)

# Register AirSim environment as a gym environment
register(
    id="learn_altitude_ctrl_discrete_env-v0", entry_point="scripts:LearnAltitudeCtrlEnvDiscrete",
)

# Register AirSim environment as a gym environment
register(
    id="learn_altitude_ctrl_fragment_env-v0", entry_point="scripts:LearnAltitudeCtrlEnvFragment",
)

# Register AirSim environment as a gym environment
register(
    id="learn_altitude_ctrl_test_env-v0", entry_point="scripts:LearnAltitudeCtrlEnvTest",
)


# Register AirSim environment as a gym environment
register(
    id="learn_path_env-v0", entry_point="scripts:LearnPathEnv",
)

# Register AirSim environment as a gym environment
register(
    id="test_env-v0", entry_point="scripts:LearnPathEnv",
)
