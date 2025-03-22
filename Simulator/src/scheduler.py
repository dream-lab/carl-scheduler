# This class schedules the containers based on the algorithm

import copy

class Scheduler(object):
    def __init__(self, env, algorithm):
        self.env = env
        self.algorithm = algorithm
        self.simulation = None
        self.cluster = None
        self.destroyed = False
        self.container_times = None
        

    def connect(self, simulation):
        self.simulation = simulation
        self.cluster = simulation.cluster

    def make_decision(self):
        while True:
            if self.cluster.can_schedule :
                returned = self.algorithm(self.cluster, self.env.now,)
                if returned == 0: 
                    break
            else:
                break

    def run(self):
        
        while not self.destroyed and not self.simulation.completed:

            self.make_decision()
            yield self.env.timeout(1)
        self.destroyed = True