import numpy as np
import math
import scipy.integrate
import time
import datetime
import threading


class Propeller:
    def __init__(self, prop_dia, prop_pitch, Motor_limits, thrust_unit='N'):
        self.dia = prop_dia
        self.pitch = prop_pitch
        self.thrust_unit = thrust_unit
        self.speed = 0  # RPM
        self.thrust: float = 0
        self.min_limit = Motor_limits[0]
        self.max_limit = Motor_limits[1]

    def set_speed(self, speed):
        if speed > self.max_limit:
            speed = self.max_limit
        if speed < self.min_limit and speed != 0:
            speed = self.min_limit
        self.speed = speed
        # From http://www.electricrcaircraftguy.com/2013/09/propeller-static-dynamic-thrust-equation.html
        self.thrust = 4.392e-8 * self.speed * math.pow(self.dia, 3.5) / (math.sqrt(self.pitch))
        self.thrust = self.thrust * (4.23e-4 * self.speed * self.pitch)
        if self.thrust_unit == 'Kg':
            self.thrust = self.thrust * 0.101972


class Quadcopter:
    def __init__(self, init: dict, gravity=9.81, b=0.0245):
        self.init = init
        self.weight = init['weight']
        self.r = self.init['r']
        self.L = self.init['L']
        self.g = gravity
        self.b = b

        self.thread_object = None
        self.ode = scipy.integrate.ode(self.state_dot).set_integrator('vode', nsteps=500, method='bdf')
        self.time = datetime.datetime.now()
        self.integrate_time = 0
        self.state = np.zeros(12)
        self.state[0:3] = self.init['position']

        # self.sim.moveToZAsync(2.5-float(self.state[2]), 1)
        time.sleep(0.5)

        self.state[6:9] = self.init['orientation']
        self.m1 = Propeller(self.init['prop_size'][0], self.init['prop_size'][1], self.init['Motor_limits'])
        self.m2 = Propeller(self.init['prop_size'][0], self.init['prop_size'][1], self.init['Motor_limits'])
        self.m3 = Propeller(self.init['prop_size'][0], self.init['prop_size'][1], self.init['Motor_limits'])
        self.m4 = Propeller(self.init['prop_size'][0], self.init['prop_size'][1], self.init['Motor_limits'])
        # From Quadrotor Dynamics and Control by Randal Beard
        ixx = ((2 * self.weight * self.r ** 2) / 5) + (
                2 * self.weight * self.L ** 2)
        iyy = ixx
        izz = ((2 * self.weight * self.r ** 2) / 5) + (
                4 * self.weight * self.L ** 2)
        self.I = np.array([[ixx, 0, 0], [0, iyy, 0], [0, 0, izz]])
        self.invI = np.linalg.inv(self.I)
        self.run = True

    @staticmethod
    def rotation_matrix(angles):
        ct = math.cos(angles[0])
        cp = math.cos(angles[1])
        cg = math.cos(angles[2])
        st = math.sin(angles[0])
        sp = math.sin(angles[1])
        sg = math.sin(angles[2])
        r_x = np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])
        r_y = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        r_z = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]])
        r = np.dot(r_z, np.dot(r_y, r_x))
        return r

    @staticmethod
    def wrap_angle(val):
        return (val + np.pi) % (2 * np.pi) - np.pi

    def state_dot(self, time, state):
        state_dot = np.zeros(12)
        # The velocities(t+1 x_dots equal the t x_dots)
        state_dot[0] = self.state[3]
        state_dot[1] = self.state[4]
        state_dot[2] = self.state[5]
        # The acceleration
        x_dotdot = np.array([0, 0, -self.weight * self.g]) + np.dot(
            self.rotation_matrix(self.state[6:9]), np.array([0, 0, (
                    self.m1.thrust + self.m2.thrust + self.m3.thrust +
                    self.m4.thrust)])) / self.weight
        state_dot[3] = x_dotdot[0]
        state_dot[4] = x_dotdot[1]
        state_dot[5] = x_dotdot[2]
        # The angular rates(t+1 theta_dots equal the t theta_dots)
        state_dot[6] = self.state[9]
        state_dot[7] = self.state[10]
        state_dot[8] = self.state[11]
        # The angular accelerations
        omega = self.state[9:12]
        tau = np.array([self.L * (self.m1.thrust - self.m3.thrust),
                        self.L * (self.m2.thrust - self.m4.thrust),
                        self.b * (self.m1.thrust - self.m2.thrust + self.m3.thrust - self.m4.thrust)])
        omega_dot = np.dot(self.invI, (tau - np.cross(omega, np.dot(self.I, omega))))
        state_dot[9] = omega_dot[0]
        state_dot[10] = omega_dot[1]
        state_dot[11] = omega_dot[2]
        return state_dot

    def update(self, dt):
        self.ode.set_initial_value(self.state, 0)
        self.state = self.ode.integrate(self.ode.t + dt)
        self.state[6:9] = self.wrap_angle(self.state[6:9])
        self.state[2] = max(0, self.state[2])
        self.integrate_time += dt

    def set_motor_speeds(self, speeds):
        self.m1.set_speed(speeds[0])
        self.m2.set_speed(speeds[1])
        self.m3.set_speed(speeds[2])
        self.m4.set_speed(speeds[3])

    def get_position(self):
        return self.state[0:3]

    def get_linear_rate(self):
        return self.state[3:6]

    def get_orientation(self):
        return self.state[6:9]

    def get_angular_rate(self):
        return self.state[9:12]

    def get_state(self):
        return self.state

    def set_position(self, position):
        self.state[0:3] = position

    def set_orientation(self, orientation):
        self.state[6:9] = orientation

    def get_time(self):
        return self.time

    def thread_run(self, dt, time_scaling):
        rate = time_scaling * dt
        last_update = self.time
        while self.run:
            time.sleep(0)
            self.time = datetime.datetime.now()
            if (self.time - last_update).total_seconds() > rate:
                self.update(dt)
                last_update = self.time

    def get_trust(self) -> list[float]:
        return [self.m1.thrust, self.m2.thrust, self.m3.thrust, self.m4.thrust]

    def start_thread(self, dt=0.002, time_scaling=1):
        self.run = True
        self.thread_object = threading.Thread(target=self.thread_run, args=(dt, time_scaling))
        self.thread_object.start()

    def stop_thread(self):
        self.run = False

    def reset(self):
        self.run = False
        if self.thread_object is not None:
            self.thread_object.join()
        self.state[0:3] = self.init['position']
        self.state[3:6] = [0, 0, 0]
        self.state[6:9] = self.init['orientation']
        self.state[9:12] = [0, 0, 0]
        self.set_motor_speeds([0, 0, 0, 0])
        self.ode = scipy.integrate.ode(self.state_dot).set_integrator('vode', nsteps=500, method='bdf')
        self.integrate_time = 0
