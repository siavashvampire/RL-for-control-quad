import numpy as np

from app.RL_logging.RL_logging import open_log_web
from app.iteration_handler.iteration_handler import IterationHandler

# asd = IterationHandler("attitude_discrete")
# asd = IterationHandler("attitude_continuous")

# asd = IterationHandler("altitude_test")
# asd = IterationHandler("path_planning")
# asd = IterationHandler("attitude_test")
#
# asd.write_flag(True)
# asd.clear()

# open_log_web()

asd = (((-1, -1, -1) ** np.random.randint(0, 2, 3)) * (0.2 - np.random.random(3) * 0.05)).squeeze()
