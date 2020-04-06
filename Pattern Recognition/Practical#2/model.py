

class Model():
    def __init__(self,method, likelihoodfncs, priors, cost_marix):
        self.method=method
        self.likelihoodfncs = likelihoodfncs
        self.priors = priors
        self.cost_matrix = cost_marix
