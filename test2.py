import numpy as np

from app.RL_logging.RL_logging import open_log_web
from app.controller.main import Controller
from app.iteration_handler.iteration_handler import IterationHandler

# asd = iterationHandler("attitude_discrete")
# asd = iterationHandler("attitude_continuous")
# asd = iterationHandler("attitude_fragment")
# asd = iterationHandler("path_planning")
# # asd = iterationHandler("attitude_test")
#
# asd.write_flag(True)
# asd.clear()
asd = [2,2,2,5]
print(np.array(asd) * 2)
# asd = open_log_web()

