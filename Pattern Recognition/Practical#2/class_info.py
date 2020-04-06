

class ClassInfo():
    """ Store information of each detected class in dataset """

    def __init__(self, label, dataset, mean_vector, covariance_matrix, total):
        self.label = label,
        self.dataset = dataset,
        self.mean_vector = mean_vector,
        self.covariance_matrix = covariance_matrix
        self.prior = len(dataset)/total
