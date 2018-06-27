import multiprocessing
from multiprocessing import Process, Manager


class ThreadManager:
	def __init__(self, thread_method, thread_callback, finished_callback, threads=0, settings = {}):
		if threads == 0:
			self.wanted_threads = multiprocessing.cpu_count()
		else:
			self.wanted_threads = threads

		self.settings = Manager().dict()
		self.settings["finished_threads"] = 0
		for key in settings.keys():
			self.settings[key] = settings[key]

		self.thread_method = thread_method
		self.thread_callback = thread_callback
		self.finished_callback = finished_callback

	def callback(self, data, settings):
		self.thread_callback(data, settings)
		settings["finished_threads"] += 1
		print("Thread " + str(settings["finished_threads"]) + " completed \n")

	def run(self, paths, data={}):
		if (len(paths) > self.wanted_threads):
			self.wanted_threads = len(paths)

		threads = []
		pathsPerThread = int(len(paths) / self.wanted_threads)
		print("Threads:", self.wanted_threads)
		print("Files:", len(paths))

		upper = 0
		for i in range(self.wanted_threads-1):
			subpaths = paths[i::self.wanted_threads]
			cpy = {**data}
			cpy["index"] = i
			p = Process(target = self.thread_method, args = (subpaths, cpy, self.settings, self.callback))
			p.start()
			threads.append(p)

		cpy = {**data}
		cpy["index"] = self.wanted_threads-1
		subpaths = paths[self.wanted_threads-1::self.wanted_threads]
		self.thread_method(subpaths, cpy, self.settings, self.callback)
		for p in threads:
			p.join()

		self.finished_callback(self.settings)