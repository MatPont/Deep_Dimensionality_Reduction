from time import time
import numpy as np
from scipy import sparse
from scipy.linalg import solve, eigh
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.utils import check_random_state
from scipy.sparse.linalg import eigsh



class LLE():
    def __init__(self, n_neighbors=5, n_components=2, n_jobs=None, verbose=True, 
                 learning_rate=0.0001, neighbors_update=False, method="direct",
                 reg=1e-3):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.n_jobs = n_jobs
        self.verbose = verbose        
        self.learning_rate = learning_rate        # Used only if method="gradient"
        self.neighbors_update = neighbors_update
        self.method = method
        self.reg = reg

        self.W = None                            # Sparse full Weight matrix [n_samples x n_samples]
        self.B = None                            # Reduced Weight matrix [n_samples x n_neighbors]
        self.lagrange_lambda = 0                 # Lambda (lagrange multiplier)
        self.knn = None                          # Store kNN result to avoid computation each time fit is called
        self.ind = None                          # Store kNN neighbors indexes

    def _update_W(self, regularization=True):
        n_samples = self.B.shape[0]
        indptr = np.arange(0, n_samples * self.n_neighbors + 1, self.n_neighbors)
        rowsum = self.B.sum(1)[:,None]
        rowsum[rowsum == 0] = 1 
        data = self.B 
        if regularization:
            data = data / rowsum
        self.W = sparse.csr_matrix((data.ravel(), self.ind.ravel(), indptr),
                      shape=(n_samples, n_samples))

    def get_W(self):
        self._update_W()
        return self.W

    def get_M(self):
        W = self.get_W()
        M = sparse.eye(*W.shape) - W
        M = M.T.dot(M)
        return M

    # From sklearn's "null_space" function
    # https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/manifold/_locally_linear.py
    def get_Y(self, k_skip=1, eigen_solver='arpack', tol=1E-6, max_iter=100,
            random_state=None):
        M = self.get_M()
        k = self.n_components
        if eigen_solver == 'auto':
            if M.shape[0] > 200 and k + k_skip < 10:
                eigen_solver = 'arpack'
            else:
                eigen_solver = 'dense'

        if eigen_solver == 'arpack':
            random_state = check_random_state(random_state)
            # initialize with [-1,1] as in ARPACK
            v0 = random_state.uniform(-1, 1, M.shape[0])
            try:
                eigen_values, eigen_vectors = eigsh(M, k + k_skip, sigma=0.0,
                                                    tol=tol, maxiter=max_iter,
                                                    v0=v0)
            except RuntimeError as msg:
                raise ValueError("Error in determining null-space with ARPACK. "
                                  "Error message: '%s'. "
                                  "Note that method='arpack' can fail when the "
                                  "weight matrix is singular or otherwise "
                                  "ill-behaved.  method='dense' is recommended. "
                                  "See online documentation for more information."
                                  % msg)

            return eigen_vectors[:, k_skip:], np.sum(eigen_values[k_skip:])
        elif eigen_solver == 'dense':
            if hasattr(M, 'toarray'):
                M = M.toarray()
            eigen_values, eigen_vectors = eigh(
                M, eigvals=(k_skip, k + k_skip - 1), overwrite_a=True)
            index = np.argsort(np.abs(eigen_values))
            return eigen_vectors[:, index], np.sum(eigen_values)
        else:
            raise ValueError("Unrecognized eigen_solver '%s'" % eigen_solver)        

    def compute_weight_loss(self, X):
        W = self.get_W()
        X_csr = sparse.csr_matrix(X)
        #return np.square(X - np.dot(W.todense(), X)).sum()
        return (X_csr - W.dot(X_csr)).power(2).sum()

    def _one_fit(self, X):
        Z = X[self.ind]
        n_samples, n_neighbors = X.shape[0], Z.shape[1]
        ones = np.ones(n_neighbors)

        for i, A in enumerate(Z.transpose(0, 2, 1)):
            #C = A.T - X[i]  
            C = X[i] - A.T
            G = np.dot(C, C.T)

            if self.method == "gradient":
                # Gradient descent way
                w_gradient = 2 * np.dot(G, self.B[i, :]) - self.lagrange_lambda * ones
                lambda_gradient = np.dot(ones, self.B[i, :]) - 1

                self.lagrange_lambda -= self.learning_rate * lambda_gradient
                self.B[i, :] -= self.learning_rate * w_gradient

                b_sum = self.B[i, :].sum()
                if b_sum != 0: 
                    self.B[i, :] = self.B[i, :] / b_sum 

            elif self.method == "direct":
                # Direct way, from sklearn's "barycenter_weights" function
                # https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/manifold/_locally_linear.py
                trace = np.trace(G)
                if trace > 0:
                    R = self.reg * trace
                else:
                    R = self.reg
                G.flat[::Z.shape[1] + 1] += R
                w = solve(G, ones, sym_pos=True)
                self.B[i, :] = w / np.sum(w)


    def fit(self, X, epoch=1):
        # Run kNN algorithm      
        if self.knn is None or self.neighbors_update:
            if self.verbose: print("Run kNN...")
            self.knn = NearestNeighbors(self.n_neighbors + 1, n_jobs=self.n_jobs).fit(X)
            self.ind = self.knn.kneighbors(X, return_distance=False)[:, 1:]
        
        # Init B matrix
        if self.B is None:
            """self.B = np.random.random((X.shape[0], self.n_neighbors))
            self.B /= self.B.sum(1)[:,None]"""
            self.B = np.zeros((X.shape[0], self.n_neighbors))

        # Train
        if self.verbose: 
            print("Train...")
            #print("LLE loss = ", self.compute_weight_loss(X))
        for ep in range(epoch):
            t0 = time()
            self._one_fit(X)
            t1 = time()
            if self.verbose:
                print(": %.2g sec" % (t1 - t0))
            if self.method != "direct": 
                if ep % 100 == 0 and self.verbose:
                    print(ep, " \ ", epoch)
                    print("LLE loss = ", self.compute_weight_loss(X))
            else:
                break

        if self.verbose:
            print("LLE loss = ", self.compute_weight_loss(X))

        return self

    def transform(self, X):
        return None

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

