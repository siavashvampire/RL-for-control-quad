import numpy as np

from app.iteration_handler.iteration_handler import IterationHandler

# asd = IterationHandler("attitude_discrete")
# asd = IterationHandler("attitude_continuous")
from app.models.quadcopter_dynamics.quadcopter import Propeller

asd = IterationHandler("altitude_test")
# asd = IterationHandler("path_planning")
# asd = IterationHandler("attitude_test")
#
asd.write_flag(True)
asd.clear()

# print(np.random.random(10))
