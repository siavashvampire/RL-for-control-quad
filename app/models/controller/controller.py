import numpy as np
import math
import time
import threading


class ControllerPID:
    target: list[int]
    yaw_target: float
    run: bool

    def __init__(self, get_state, get_time, actuate_motors, params):

        self.actuate_motors = actuate_motors
        self.get_state = get_state
        self.get_time = get_time
        self.MOTOR_LIMITS = params['Motor_limits']
        self.TILT_LIMITS = [(params['Tilt_limits'][0] / 180.0) * 3.14, (params['Tilt_limits'][1] / 180.0) * 3.14]
        self.YAW_CONTROL_LIMITS = params['Yaw_Control_Limits']
        self.Z_LIMITS = [self.MOTOR_LIMITS[0] + params['Z_XY_offset'], self.MOTOR_LIMITS[1] - params['Z_XY_offset']]
        self.LINEAR_P = params['Linear_PID']['P']
        self.LINEAR_I = params['Linear_PID']['I']
        self.LINEAR_D = params['Linear_PID']['D']
        self.LINEAR_TO_ANGULAR_SCALER = params['Linear_To_Angular_Scaler']
        self.YAW_RATE_SCALER = params['Yaw_Rate_Scaler']
        self.ANGULAR_P = params['Angular_PID']['P']
        self.ANGULAR_I = params['Angular_PID']['I']
        self.ANGULAR_D = params['Angular_PID']['D']
        self.xi_term = 0
        self.yi_term = 0
        self.zi_term = 0
        self.I_x = 0
        self.I_y = 0
        self.I_z = 0
        self.thetai_term = 0
        self.phii_term = 0
        self.gammai_term = 0
        self.I_theta = 0
        self.I_phi = 0
        self.I_gamma = 0
        self.thread_object = None
        self.target = [0, 0, 0]
        self.yaw_target = 0.0
        self.run = True

    @staticmethod
    def wrap_angle(val):
        return (val + np.pi) % (2 * np.pi) - np.pi

    def update(self):
        [dest_x, dest_y, dest_z] = self.target
        [x, y, z, x_dot, y_dot, z_dot, theta, phi, gamma, theta_dot, phi_dot, gamma_dot] = self.get_state()

        x_error = dest_x - x
        y_error = dest_y - y
        z_error = dest_z - z

        self.I_x += x_error
        self.I_y += y_error
        self.I_z += z_error

        self.xi_term += self.LINEAR_I[0] * x_error
        self.yi_term += self.LINEAR_I[1] * y_error
        self.zi_term += self.LINEAR_I[2] * z_error

        dest_x_dot = self.LINEAR_P[0] * x_error + self.LINEAR_D[0] * (-x_dot) + self.xi_term
        dest_y_dot = self.LINEAR_P[1] * y_error + self.LINEAR_D[1] * (-y_dot) + self.yi_term
        dest_z_dot = self.LINEAR_P[2] * z_error + self.LINEAR_D[2] * (-z_dot) + self.zi_term
        # print("dest_z_dot =", dest_z_dot)
        throttle = np.clip(dest_z_dot, self.Z_LIMITS[0], self.Z_LIMITS[1])

        dest_theta = self.LINEAR_TO_ANGULAR_SCALER[0] * (dest_x_dot * math.sin(gamma) - dest_y_dot * math.cos(gamma))
        dest_phi = self.LINEAR_TO_ANGULAR_SCALER[1] * (dest_x_dot * math.cos(gamma) + dest_y_dot * math.sin(gamma))
        dest_gamma = self.yaw_target
        dest_theta, dest_phi = np.clip(dest_theta, self.TILT_LIMITS[0], self.TILT_LIMITS[1]), \
                               np.clip(dest_phi, self.TILT_LIMITS[0], self.TILT_LIMITS[1])

        theta_error = dest_theta - theta
        phi_error = dest_phi - phi

        gamma_dot_error = (self.YAW_RATE_SCALER * self.wrap_angle(dest_gamma - gamma)) - gamma_dot

        self.I_theta += theta_error
        self.I_phi += phi_error
        self.I_gamma += gamma_dot_error

        self.thetai_term += self.ANGULAR_I[0] * theta_error
        self.phii_term += self.ANGULAR_I[1] * phi_error
        self.gammai_term += self.ANGULAR_I[2] * gamma_dot_error

        x_val = self.ANGULAR_P[0] * theta_error + self.ANGULAR_D[0] * (-theta_dot) + self.thetai_term
        y_val = self.ANGULAR_P[1] * phi_error + self.ANGULAR_D[1] * (-phi_dot) + self.phii_term
        z_val = self.ANGULAR_P[2] * gamma_dot_error + self.gammai_term
        z_val = np.clip(z_val, self.YAW_CONTROL_LIMITS[0], self.YAW_CONTROL_LIMITS[1])

        m1 = throttle + x_val + z_val
        m2 = throttle + y_val - z_val
        m3 = throttle - x_val + z_val
        m4 = throttle - y_val - z_val

        M = np.clip([m1, m2, m3, m4], self.MOTOR_LIMITS[0], self.MOTOR_LIMITS[1])
        self.actuate_motors(M)

    def update_target(self, target):
        self.target = target

    def update_target_rel(self, target):
        self.target = [m + n for m, n in zip(self.target, target)]

    def update_yaw_target(self, target):
        self.yaw_target = self.wrap_angle(target)

    def thread_run(self, update_rate, time_scaling):
        update_rate = update_rate * time_scaling
        last_update = self.get_time()
        while self.run:
            time.sleep(0)
            self.time = self.get_time()
            if (self.time - last_update).total_seconds() > update_rate:
                self.update()
                last_update = self.time

    def start_thread(self, update_rate=0.005, time_scaling=1):
        self.run = True
        self.thread_object = threading.Thread(target=self.thread_run, args=(update_rate, time_scaling))
        self.thread_object.start()

    def stop_thread(self):
        self.run = False

    def get_diff_linear(self) -> (np.float64, np.float64):
        [dest_x, dest_y, dest_z] = self.target
        [x, y, z, x_dot, y_dot, z_dot, _, _, _, _, _, _] = self.get_state()
        x_error = dest_x - x
        y_error = dest_y - y
        z_error = dest_z - z
        return np.linalg.norm([x_error, y_error, z_error]), np.linalg.norm([x_dot, y_dot, z_dot])

    def get_error_linear(self) -> np.ndarray:
        [dest_x, dest_y, dest_z] = self.target
        [x, y, z, _, _, _, _, _, _, _, _, _] = self.get_state()
        x_error = dest_x - x
        y_error = dest_y - y
        z_error = dest_z - z

        return np.array([x_error, y_error, z_error])

    def get_diff_angular(self) -> (np.float64, np.float64):
        [dest_x, dest_y, _] = self.target
        [x, y, _, x_dot, y_dot, _, theta, phi, gamma, theta_dot, phi_dot, gamma_dot] = self.get_state()

        x_error = dest_x - x
        y_error = dest_y - y

        dest_x_dot = self.LINEAR_P[0] * x_error + self.LINEAR_D[0] * (-x_dot) + self.xi_term
        dest_y_dot = self.LINEAR_P[1] * y_error + self.LINEAR_D[1] * (-y_dot) + self.yi_term

        dest_theta = self.LINEAR_TO_ANGULAR_SCALER[0] * (dest_x_dot * math.sin(gamma) - dest_y_dot * math.cos(gamma))
        dest_phi = self.LINEAR_TO_ANGULAR_SCALER[1] * (dest_x_dot * math.cos(gamma) + dest_y_dot * math.sin(gamma))
        dest_gamma = self.yaw_target
        dest_theta, dest_phi = np.clip(dest_theta, self.TILT_LIMITS[0], self.TILT_LIMITS[1]), \
                               np.clip(dest_phi, self.TILT_LIMITS[0], self.TILT_LIMITS[1])

        gamma_error = self.wrap_angle(dest_gamma - gamma)
        theta_error = dest_theta - theta
        phi_error = dest_phi - phi
        return np.linalg.norm([theta_error, phi_error, gamma_error]), np.linalg.norm([theta_dot, phi_dot, gamma_dot])

    def set_LINEAR_PID(self, params) -> None:
        self.LINEAR_P = params[0]
        self.LINEAR_I = params[1]
        self.LINEAR_D = params[2]

    def set_ANGULAR_PID(self, params) -> None:
        self.ANGULAR_P = params[0]
        self.ANGULAR_I = params[1]
        self.ANGULAR_D = params[2]

    def get_ANGULAR_PID(self) -> np.ndarray:
        return np.array([self.ANGULAR_P, self.ANGULAR_I, self.ANGULAR_D])

    def get_LINEAR_PID(self) -> np.ndarray:
        return np.array([self.LINEAR_P, self.LINEAR_I, self.LINEAR_D])

    def reset(self):
        self.run = False
        if self.thread_object is not None:
            self.thread_object.join()
        self.xi_term = 0
        self.yi_term = 0
        self.zi_term = 0
        self.I_x = 0
        self.I_y = 0
        self.I_z = 0
        self.thetai_term = 0
        self.phii_term = 0
        self.gammai_term = 0
        self.I_theta = 0
        self.I_phi = 0
        self.I_gamma = 0
        self.target = [0, 0, 0]
        self.yaw_target = 0.0
        self.set_ANGULAR_PID([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        self.set_LINEAR_PID([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    def get_obs(self):
        [dest_x, dest_y, dest_z] = self.target
        [x, y, z, x_dot, y_dot, z_dot, theta, phi, gamma, theta_dot, phi_dot, gamma_dot] = self.get_state()

        x_error = dest_x - x
        y_error = dest_y - y
        z_error = dest_z - z

        dest_x_dot = self.LINEAR_P[0] * x_error + self.LINEAR_D[0] * (-x_dot) + self.xi_term
        dest_y_dot = self.LINEAR_P[1] * y_error + self.LINEAR_D[1] * (-y_dot) + self.yi_term

        dest_theta = self.LINEAR_TO_ANGULAR_SCALER[0] * (dest_x_dot * math.sin(gamma) - dest_y_dot * math.cos(gamma))
        dest_phi = self.LINEAR_TO_ANGULAR_SCALER[1] * (dest_x_dot * math.cos(gamma) + dest_y_dot * math.sin(gamma))
        dest_gamma = self.yaw_target
        dest_theta, dest_phi = np.clip(dest_theta, self.TILT_LIMITS[0], self.TILT_LIMITS[1]), \
                               np.clip(dest_phi, self.TILT_LIMITS[0], self.TILT_LIMITS[1])

        theta_error = dest_theta - theta
        phi_error = dest_phi - phi

        gamma_dot_error = (self.YAW_RATE_SCALER * self.wrap_angle(dest_gamma - gamma)) - gamma_dot

        return [[x_error, x_dot, self.I_x],
                [y_error, y_dot, self.I_y],
                [z_error, z_dot, self.I_z]], [[theta_error, theta_dot, self.I_theta],
                                              [phi_error, phi_dot, self.I_phi],
                                              [gamma_dot_error, gamma_dot_error, self.I_gamma]]
