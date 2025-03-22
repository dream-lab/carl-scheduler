# This class will supply container requests one by one  
class Broker(object):
 
    def __init__(self, environment, containers, containers_dict):
        self.env = environment
        self.containers = containers
        self.containers_dict = containers_dict
        self.cluster = None 
        self.simulation = None 
        self.completed = False 

    def connect(self, simulation):
        self.simulation = simulation
        self.cluster = simulation.cluster

    # Since Broker will be a thread created by Simpy, we define a run method for it.
    def run(self):
        for container in self.containers:
            
            cnt_idx = container[0].astype(int)
            assert container[1].astype(int) >= self.env.now
            yield self.env.timeout(container[1].astype(int) - self.env.now)
            # print('Task arrived for cid %s at time %f with duration %f' % (self.containers_dict[cnt_idx], self.env.now, container[6]))
            
            self.cluster.add_container(container, self.containers_dict[cnt_idx])

        self.completed = True
        
