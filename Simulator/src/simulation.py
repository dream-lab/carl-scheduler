class Simulation(object):

    def __init__(self, env, cluster, broker, scheduler, static_ct, **kwargs):
        self.env = env
        self.cluster = cluster
        self.broker = broker
        self.scheduler = scheduler

        self.broker.connect(self)
        self.scheduler.connect(self)
        
        self.static_ct = static_ct


    def run(self):
        self.env.process(self.broker.run())
        self.env.process(self.scheduler.run())

    @property
    def completed(self):
        return self.broker.completed and len(self.cluster.container_which_has_waiting_instance) == 0