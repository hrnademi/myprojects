import numpy as np


def pca(samples):
    def propose_suitable_d(eigenvalues):
        """ Propose a suitable d using POV = 95% """
        sum_D = sum(eigenvalues)
        for d in range(0, len(eigenvalues)):
            pov = sum(eigenvalues[:d])/sum_D
            if pov > 0.95:
                return d

    def pca_formula(d, train_data, mean_vector, eigenvectors):
        x_bar = train_data-np.array([mean_vector])
        eigenvectors = eigenvectors.T
        w = eigenvectors[:d]

        # PCA formula
        y = np.matmul(a=w, b=x_bar.transpose()).transpose()

        return y

    # Compute mean vector of train data
    mean_vector = np.mean(samples, axis=0)

    # Compute covariance matrix of train data
    covariance_matrix = np.cov(
        np.asarray(samples).transpose(),
        rowvar=True
    )

    # Compute eigenvector and eigenvalue of covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Sort eigenvalues and eigenvectors
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Compute eigenvalue of covariance matrix
    eigenvalues = np.linalg.eigvals(covariance_matrix)

    # suitable_d = propose_suitable_d(eigenvalues)
    suitable_d = 2

    # PCA
    samples_pca = pca_formula(
        d=suitable_d,
        train_data=samples,
        mean_vector=mean_vector,
        eigenvectors=eigenvectors
    )

    return samples_pca
