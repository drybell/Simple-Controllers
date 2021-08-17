import cv2 
import numpy as np
import random
from abc import ABCMeta, abstractmethod
from time import sleep 
import sys
import argparse

### TODO: 3D version with the robot now having a viewing fulcrum 
###		  bind it to a ground plane, or even try out an aerial approach 

### THOUGHTS ON ABOVE
#
#	instead of simulating the robot's movement with absolute knowledge of the 
# 	target's position, using the viewing fulcrum, we can simulate what 
#	the robot "sees" and then build a simulated control algorithm that 
# 	attempts to maintain the target in the center of the robot's viewing 
# 	window. Start this with only Z rotation, then move to more degrees of freedom

### TODO: THREEJS BABY after 2d exploration is done !!!

def unit_vector(vector):
	""" Returns the unit vector of the vector"""
	return vector / np.linalg.norm(vector)

def get_angle(vector1, vector2):
	""" Returns the angle in radians between given vectors"""
	v1_u = unit_vector(vector1)
	v2_u = unit_vector(vector2)
	cross = np.cross(v1_u, v2_u)
	dot   = np.dot(cross, [0, 0, 1])
	# print(dot)
	dot_p = np.dot(v1_u, v2_u)
	# dot_p = min(max(dot_p, -1.0), 1.0)
	# print(f'{sign * np.arccos(dot_p)} {cross}')
	angle = np.arccos(dot_p)

	angle = -angle if dot[-1] < 0 else angle
	if angle > 3.14159:
		angle = -1 * (2 * 3.14159 - angle)
	elif angle < -3.14159:
		angle = -1 * (2 * 3.14159 + angle)

	return 0 if np.isnan(angle) else angle

class ArgumentNotFoundException(Exception):
	def __init__(self, message):            
		# Call the base class constructor with the parameters it needs
		super().__init__(message)

class Object:
	__metaclass__ = ABCMeta

	def __init__(self):
		return

	@abstractmethod
	def update(self):
		return

	@abstractmethod
	def show(self):
		return

	@abstractmethod
	def type_of(self):
		return

class Robot(Object): 

	## Sets the robot's params and displays on the screen
	def __init__(self, x, y, controller='p', facing_vector=[1,0], color=[255,0,0]):
		self.x = x
		self.y = y 
		self.velocity   = 0
		self.rotation   = 0 
		self.controller = controller
		self.size = 50
		self.color = color
		self.sprite     = [[int(x + (self.size / 2)), int(y + (self.size / 2))], int(self.size / 2), self.color, -1]
		facing_mag = np.linalg.norm(facing_vector)
		if (facing_mag > 1):
			self.head = np.array(facing_vector) / facing_mag
		else:
			self.head = np.array(facing_vector)
		if controller   == 'p':
			self.update = self.p_control		
		elif controller == 'pd':
			self.update = self.pd_control
		elif controller  == 'pid':
			self.update = self.pid_control
		elif controller == 'pi':
			self.update = self.pi_control
		elif controller == 'pd-special':
			self.update = self.pd_special
		else: 
			raise ArgumentNotFoundException(f"{controller} is not a valid controller")
		start_points = np.array([int(x + (self.size / 2)) , int(y + (self.size / 2))])
		end_points   = start_points + (self.head * 50)
		end_points   = [int(l) for l in end_points]
		self.line    = start_points, end_points, (255, 255, 255), 2 
		self.K_P_d   = .1
		self.K_P_a   = .25
		self.K_D     = .1
		self.K_I     = .075
		self.prev_x  = x
		self.prev_y  = y
		self.a_error = 0
		self.d_error = 0
		self.sum_a_error = 0
		self.sum_d_error = 0
		self.max_velocity = 10

	def p_control(self, target): 
		# distance and angle error 
		angle_error = self.theta_to_target(target)
		# print(self.theta_to_target(target) / 3.14159 * 180)
		dist_error  = self.distance_to_target(target)
		new_vel = self.K_P_d * dist_error
		self.velocity = min(new_vel, self.max_velocity)
		self.rotation = self.K_P_a * angle_error
		self.move()

	def pd_special(self, target):
		angle_error = self.theta_to_target(target)
		# print(self.theta_to_target(target) / 3.14159 * 180)
		dist_error  = self.distance_to_target(target)
		self.rotation    = self.K_P_a * angle_error + ((self.K_D * self.a_error) / 60)
		new_vel = self.K_P_d * dist_error + ((self.K_D * self.d_error) / 60)
		self.velocity = min(new_vel, self.max_velocity)
		self.a_error = angle_error
		self.d_error = dist_error
		self.move()

	def move(self):
		# update heading vector 
		rot_vec = np.array([[np.cos(self.rotation), - np.sin(self.rotation)],
				   [np.sin(self.rotation), np.cos(self.rotation)]])

		self.head = np.dot(np.array(self.head), rot_vec) / np.linalg.norm(np.dot(np.array(self.head), rot_vec))
		vel_vec   = self.velocity * self.head
		
		self.x = self.x + vel_vec[0]
		self.y = self.y + vel_vec[1]

		self.sprite     = [[int(self.x + (self.size / 2)), int(self.y + (self.size / 2))], int(self.size / 2), self.color, -1]
		start_points = np.array([int(self.x + (self.size / 2)) , int(self.y + (self.size / 2))])
		end_points   = start_points + (self.head * 50)
		end_points   = [int(l) for l in end_points]
		self.line    = start_points, end_points, (255, 255, 255), 2 

	def pd_control(self, target): 
		angle_error = self.theta_to_target(target)
		# print(self.theta_to_target(target) / 3.14159 * 180)
		dist_error  = self.distance_to_target(target)
		self.rotation    = self.K_P_a * angle_error + (self.K_D * self.a_error)
		new_vel = self.K_P_d * dist_error + (self.K_D * self.d_error)
		self.velocity = min(new_vel, self.max_velocity)
		self.a_error = angle_error
		self.d_error = dist_error
		self.move()

	def pid_control(self, target):
		angle_error = self.theta_to_target(target)
		# print(self.theta_to_target(target) / 3.14159 * 180)
		dist_error  = self.distance_to_target(target)
		self.sum_a_error += angle_error
		self.sum_d_error += dist_error
		self.rotation    = self.K_P_a * angle_error + (self.K_D * self.a_error) + (self.K_I * self.sum_a_error)
		new_vel = self.K_P_d * dist_error + (self.K_D * self.d_error) + (self.K_I * self.sum_d_error)
		self.velocity = min(new_vel, self.max_velocity)
		self.a_error = angle_error
		self.d_error = dist_error
		self.move()

	def pi_control(self, target):
		angle_error = self.theta_to_target(target)
		# print(self.theta_to_target(target) / 3.14159 * 180)
		dist_error  = self.distance_to_target(target)
		self.sum_a_error += angle_error
		self.sum_d_error += dist_error
		self.rotation    = self.K_P_a * angle_error + (self.K_I * self.sum_a_error)
		new_vel = self.K_P_d * dist_error + (self.K_I * self.sum_d_error)
		self.velocity = min(new_vel, self.max_velocity)
		self.move()

	def debug(self):
		debug_str = "ROBOT:"
		debug_str += f"\tCOORDS:   ({self.x}, {self.y})\n"
		debug_str += f"\tVELOCITY: {self.velocity} p/s\n"
		debug_str += f"\tHEADING:  {self.head}\n"
		print(debug_str)

	def type_of(self):
		return "ROBOT"

	def set_id(self, id_):
		self.id = id_

	def get_id(self):
		if self.id is not None: 
			return self.id

	def theta_to_target(self, target): 
		v0 = [self.x + (self.size / 2), self.y + (self.size / 2)]
		v1 = np.array([target.x, target.y]) 
		vec_to_target = v1 - v0 
		# angle = np.arctan2(np.cross(v0,v1), np.dot(v0,v1))
		# return np.arctan2(v1[1] - v0[1], v1[0] - v0[0])
		# angle = np.arctan2(np.linalg.det([v1,v0]),np.dot(v1,v0))
		angle = get_angle(vec_to_target, self.head)
		# angle = np.arccos(np.clip(np.dot(v1 / np.linalg.norm(v1), v0 / np.linalg.norm(v0)), -1.0, 1.0))
		# angle = np.arctan2(v1[1] - v0[1], v1[0] - v0[0])
		# print(angle / M_PI * 180)
		# print(angle)
		return angle

	def distance_to_target(self, target):
		# print(np.sqrt((self.y - target.y)**2 + (self.x - target.x)**2))
		dist = np.sqrt(((self.y + (self.size / 2)) - target.y)**2 + ((self.x + (self.size / 2)) - target.x)**2)
		# print(dist)
		if dist < 50:
			target.update()
		return dist	


## Sets the target's params and displays on the screen 
class Target(Object): 

	def __init__(self, x, y, size):
		self.x = (x + (size / 2))
		self.y = (y + (size / 2))
		self.size = size
		self.color = [0, 247, 255]
		# self.sprite = np.zeros((size, size, 3), dtype='uint8')
		# self.sprite[:] = self.color
		self.sprite = [[int(x + (size / 2)), int(y + (size / 2))], int(size / 2), self.color, -1]

	def debug(self):
		debug_str = "TARGET"
		debug_str += f"\tCOORDS: ({self.x}, {self.y})\n"
		print(debug_str)

	def update(self):
		self.x = random.randint(0,1000) + (self.size / 2)
		self.y = random.randint(0,1000) + (self.size / 2)
		self.sprite = [[int(self.x), int(self.y)], int(self.size / 2), self.color, -1]

	def type_of(self): 
		return "TARGET"

	def set_id(self, id_):
		self.id = id_

	def get_id(self):
		if self.id is not None: 
			return self.id

class Window:

	def __init__(self, width, height, hz, color='black'): 
		self.width  = width
		self.height = height
		self.refresh_rate = hz
		color_table = {'black': [0, 0, 0]} 
		if isinstance(color, str):
			color = color_table[color]
		else: 
			color = color
		# TODO: add other colors and such by transforming a 2d array to size (w, h, 3) and 
		# 	    filling by color 
		self.window = np.zeros((height, width, 3), dtype = "uint8")
		self.objects = {}
		self.id_ctr  = 0

	def paint(self):
		# draw objects on background
		# Foreach object that's tracked, use the object's sprite data to 
		# paint its location onto the background
		for key in self.objects.keys():
			val = self.objects[key]
			if (val.type_of() == "TARGET"):
				self.window = cv2.circle(self.window, *val.sprite)
			elif (val.type_of() == "ROBOT"):
				self.window = cv2.circle(self.window, *val.sprite)
				self.window = cv2.line(self.window, *val.line)
			else: 
				print(f"WARNING: UNKNOWN OBJECT {val.get_id()} IS NOT BEING DRAWN")
		cv2.imshow("TEST", self.window)
		self.window = np.zeros((self.height, self.width, 3), dtype = "uint8")

	def update(self):
		while True:
			for key in self.objects.keys():
				val = self.objects[key]
				if (val.type_of() == "ROBOT"):
					# get the target object 
					target = [self.objects[k] for k in self.objects.keys() if self.objects[k].type_of() == "TARGET"][0]
					val.update(target)

			self.paint()
			cv2.waitKey(self.refresh_rate)
		# sleep(self.refresh_rate)

	def add_object(self, obj):
		# assert object is an Object class
		if not isinstance(obj, Object):
			raise Exception("Can't pass in object to window")
		else: 
			id_ = self.create_id()
			obj.set_id(id_)
			self.objects.update({id_: obj})

	def remove_object(self, key):
		if key in self.objects.keys():
			self.objects.pop(key, None)

	def debug(self):
		return

	def create_id(self):
		temp = self.id_ctr
		self.id_ctr += 1
		return str(temp)

# parser = argparse.ArgumentParser(description="Proportional Controller")
# parser.add_argument('')

# Create a window with width, height, and refresh rate to host sprites
window = Window(1000, 1000, 20)

# create a target for robots to travel towards
target = Target(50, 50, 30)

# Build robots with varying controllers 
robot  = Robot(100, 200, controller='p', facing_vector=[1,0], color=[0, 0, 255])
robot2 = Robot(350, 300, controller='pd', facing_vector=[-1,0], color=[0, 255, 0])
robot3 = Robot(500, 899, controller='pid', facing_vector=[3,2], color=[255, 0, 222])
robot4 = Robot(200, 250, controller='pi', facing_vector=[0, -1], color=[255, 0, 0])
robot5 = Robot(50, 50, controller='pd-special', facing_vector=[0, 1], color=[255,255,255])

# add objects to window 
window.add_object(robot)
window.add_object(robot2)
window.add_object(robot3)
window.add_object(robot4)
window.add_object(robot5)
window.add_object(target)

# start the playback loop of the window with sprites loaded in 
window.update()	

# remove an object from the window 
window.remove_object(robot.get_id())

# Ctrl-C to exit, exiting other ways under construction 