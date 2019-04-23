#!/usr/bin/python
import time

class SaveData:
	def __init__(self, scores=[], global_t=0, wall_t=0, start_time=0):
		self.start_time = start_time
		self.global_t = global_t
		self.wall_t = wall_t
		self.scores = scores

		self.saveRequested = False

	def requestSave(self):
		self.saveRequested = True

	def setScores(self, scores):
		self.scores = scores

	def append(self, value, pi):
		pi = [float('%.6f' % val) for val in pi]
		self.scores.append({"time": time.time(), "global": self.global_t, "value": value, "policy": pi})