import sys 
sys.path.append('../')
import simpy 
from Simulator.src.cluster import Cluster 
from Simulator.src.scheduler import Scheduler
from Simulator.src.broker import Broker
from Simulator.src.simulation import Simulation 


class Episode(object):

    def __init__(self, machines, machines_dict, containers, containers_dict, algorithm, static_ct, **kwargs):

        self.env = simpy.Environment()
        
        bins  = None
        if 'bins' in kwargs:
            bins = kwargs['bins']
        cluster = Cluster(self.env, bins)
        cluster.add_machines(machines, machines_dict)

        broker = Broker(self.env, containers, containers_dict)
        scheduler = Scheduler(self.env, algorithm)

        self.simulation = Simulation(self.env, cluster, broker, scheduler, static_ct, **kwargs)
    
    def run(self):
        self.simulation.run()
        self.env.run()