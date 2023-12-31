# TODO: write as class
import numpy as np

# State space representation from observation
def state_space(Y: np.array, p: int, P: np.array) -> np.array:
    """
    Construct state space representation of process from observation Y up to order p-1.
    :param Y: MCAR process observation, (d, N+1) np.array
    :param p: state space order, int
    :param delta_t: sampling rate of observations, float
    :param P: partition over which to simulate the MCAR process [0=t_0, t_1, ..., t_N = T], (N+1,) np.array
    :return X: state space representation (first d entries are Y), (pd, N+1) np.array
    """
    # get dimensions
    d, N = Y.shape
    X = np.zeros([p*d, N])
    X[:d, :] = Y
    # differentiate p-1 times to get state space representation
    for i in range(1, p):
        X[d*i:d*(i+1), :-1] = np.diff(X[d*(i-1):d*i, :], axis=1)/(P[1:] - P[:-1])
    return X

# Estimate MCAR parameter AA
def estimate_MCAR(Y: np.array, p: int, P: np.array, Q: np.array, b: np.array, nu: np.array, with_cov: bool = False, Sigma: np.array = None) -> np.array:
    """
    Estimate MCAR parameters from the state space realisation of a GrCAR model.
    with triplet (b, Sigma, rate*jump_F).
    :param Y: MCAR process observation, (d, N+1) np.array
    :param p: MCAR parameter, int
    :param P: finer partition over which we observe the MCAR process [0 = s_0, ..., s_N = t], (N+1,) np.array
    :param Q: coarser partition over which to approximate the integrals [0 = u_0, ..., u_M = T], (M+1,) np.array
    :param b: drift of Levy process, (d,) np.array
    :param nu: thresholding sequence, (d, M) np.array
    :param with_cov: whether to return the vectorized estimator with its estimated covariance, bool
    :param Sigma: covariance matrix of the driving Levy process - only needed if with_cov is True, (d, d) np.array
    :return if with_cov is True:
                vec_AA_hat: vectorized estimated MCAR parameters, (pd**2,) np.array
                HH: estimated covariance of the estimated MCAR parameters, (pd**2, pd**2) np.array
            else:
                AA_hat: estimated MCAR parameters, list of p (d, d) np.arrays
    """
    # get dimension
    d = Y.shape[0]

    if with_cov and Sigma is not None:
        # compute inverse of Sigma
        Sigma_inv = np.linalg.inv(Sigma)
    elif with_cov and Sigma is None:
        # estimate Sigma from data
        AA_hat = estimate_MCAR(Y, p, P, Q, b, nu, with_cov=False)
        DeltaL = recover_BDLP(Y, p, P, Q, AA_hat)
        Sigma_hat = estimate_Sigma_L(DeltaL, Q, b)
        Sigma_inv = np.linalg.inv(Sigma_hat)
    else:
        # no need to use Sigma if cov is not needed
        Sigma_inv = np.eye(d)

    # approximate the state space representation by finite differences over P
    X = state_space(Y, p, P)

    # approximate integrals on Q
    indices = [False] * len(P)
    j = 0
    for i, s in enumerate(P):
        if s == Q[j]:
            indices[i] = True
            j += 1
    X = X[:, indices]
    x0 = X[:, 0]

    # get P_0-continuous martingale part of last d entries of state space representation, i.e. of the p-1-th derivative of MCAR realisation
    DY = np.diff(X[-d:, :] - x0[-d:].reshape(-1, 1) - b.reshape(-1, 1) * Q.reshape(1, -1))

    # if nu is not provided choose from data
    if nu is None:
        nu = np.zeros_like(DY)
        for i in range(d):
            nu[i, :] = choose_nu(DY[i:i+1, :], Q, b[i])
    
    # get thresholded increments
    DY_thresh = DY * (np.abs(DY) < nu)

    # compute H
    H = np.zeros(p*d**2)
    for k in range(p):
        for i in range(d):
            integral = np.zeros((d,))
            # (i+1)-th entry of D^{p-k}Y
            integrand = X[-(k+1)*d+i, :]
            for j in range(d):
                # left-hand Riemann sum (converges in L2 to Ito Integral)
                integral[i] = np.sum(integrand[:-1] * DY_thresh[j, :])
            H[k*d**2+i*d:k*d**2+(i+1)*d] = - np.matmul(Sigma_inv, integral).reshape(-1)

    # compute [H]
    HH = np.zeros((p*d**2, p*d**2))
    for k in range(p):
        for i in range(d):
            for l in range(p):
                for j in range(d):
                    # (i+1)-th entry of D^{p-k}Y
                    integrand_one = X[-(k+1)*d+i, :]
                    # (j+1)-th entry of D^{p-l}Y
                    integrand_two = X[-(l+1)*d+j, :]
                    # left-hand Riemann sum (converges a.s. to Riemann Integral, for any choice of mid-point)
                    integral = np.sum(integrand_one[:-1] * integrand_two[:-1] * np.diff(Q))
                    HH[k*d**2+i*d:k*d**2+(i+1)*d, l*d**2+j*d:l*d**2+(j+1)*d] = Sigma_inv * integral

    # compute vec_AA_hat
    vec_AA_hat = np.matmul(np.linalg.inv(HH), H)
    
    if with_cov and Sigma is not None:
        return vec_AA_hat, HH
    elif with_cov and Sigma is None:
        return vec_AA_hat, HH, Sigma_hat
    else:
        AA_hat = []
        for i in range(p):
            AA_hat.append(vec_AA_hat[i*d**2:(i+1)*d**2].reshape((d, d)).T)
        return AA_hat

# Estimate grCAR parameter theta from state space observation
def estimate_grCAR(Y: np.array, A: np.array, p: int, P: np.array, Q: np.array, b: np.array, nu: np.array, Sigma: np.array, with_cov: bool = False) -> np.array:
    """
    Estimate grCAR parameters from the state space realisation of a grCAR model and corresponding driving Levy process (with finite Levy measure)
    with triplet (b, Sigma, rate*jump_F). Return the whole state space (CAR path is the first d entries).
    :param Y: MCAR process observation, (d, N+1) np.array
    :param A: graph adjacency matrix, (d, d) np.array
    :param p: GrCAR parameter, int
    :param P: finer partition over which we observe the GrCAR process [0 = s_0, ..., s_N = t], (N+1,) np.array
    :param Q: coarser partition over which to approximate the integrals [0 = u_0, ..., u_M = T], (M+1,) np.array
    :param b: drift of Levy process, (d,) np.array
    :param nu: thresholding sequence, (d, M) np.array
    :param Sigma: covariance matrix of the driving Levy process, (d, d) np.array
    :param with_cov: whether to return the vectorized estimator with its estimated covariance, bool
    :return if with_cov is True:
                vec_theta_hat: vectorized estimated grCAR parameters, (2p,) np.array
                KK: estimated covariance of the estimated grCAR parameters, (2p, 2p) np.array
            else:
                theta_hat: estimated grCAR parameters, (p, 2) np.array
    """
    # approximate the state space representation by finite differences over P
    X = state_space(Y, p, P)

    # approximate integrals on Q
    indices = [False] * len(P)
    j = 0
    for i, s in enumerate(P):
        if s == Q[j]:
            indices[i] = True
            j += 1
    X = X[:, indices]

    # get dimensions and time horizon
    d = len(b)
    p = int(X.shape[0]/d)
    x0 = X[:, 0]

    # compute normalised adjacency matrix
    barA = np.matmul(A, np.diag(1/np.sum(A, axis=0)))
    Sigma_inv = np.linalg.inv(Sigma)

    # get P_0-continuous martingale part of last d entries of state space representation, i.e. of the p-1-th derivative of grCAR realisation
    DY = np.diff(X[-d:, :] - x0[-d:].reshape(-1, 1) - b.reshape(-1, 1) * Q.reshape(1, -1))

    # get thresholded increments
    DY_thresh = DY * (np.abs(DY) < nu)

    # compute K
    K = np.zeros(2*p)
    for k in range(p):
        integral = np.zeros((d, d))
        if k == 0:
            integrand = X[-d:, :]
        else:
            integrand = X[-d-k*d:-k*d, :]
        for i in range(d):
            for j in range(d):
                # left-hand Riemann sum (converges in L2 to Ito Integral)
                integral[i, j] = np.sum(integrand[i, :-1] * DY_thresh[j, :])
        K[2*k + 1] = - np.sum(np.multiply(np.matmul(barA, Sigma_inv), integral))

    # compute [K]
    KK = np.zeros((2*p, 2*p))
    for k in range(p):
        for l in range(p):
            if k == 0:
                integrand_one = X[-d:, :]
            else:
                integrand_one = X[-d-k*d:-k*d, :]
            if l == 0:
                integrand_two = X[-d:, :]
            else:
                integrand_two = X[-d-l*d:-l*d, :]
            integral = np.zeros((d, d))
            for i in range(d):
                for j in range(d):
                    # left-hand Riemann sum (converges a.s. to Riemann Integral, for any choice of mid-point)
                    integral[i, j] = np.sum(integrand_one[i, :-1] * integrand_two[j, :-1] * (Q[1:] - Q[:-1]))
            KK[2*k, 2*l] = np.sum(np.multiply(Sigma_inv, integral))
            KK[2*k+1, 2*l] = np.sum(np.multiply(np.matmul(barA, Sigma_inv), integral))
            KK[2*k, 2*l+1] = np.sum(np.multiply(np.matmul(Sigma_inv, barA.T), integral))
            KK[2*k+1, 2*l+1] = np.sum(np.multiply(np.matmul(np.matmul(barA, Sigma_inv), barA.T), integral))

    # compute and return theta_hat
    vec_theta_hat = np.matmul(np.linalg.inv(KK), K)
    if with_cov:
        return vec_theta_hat, KK
    else:
        return vec_theta_hat.reshape((p, 2))

def recover_BDLP(Y: np.array, p: int, P: np.array, Q: np.array, AA: list):
    """
    Recover the increments of the background driving Levy process L from MCAR observation Q and parameters AA.
    First recover increments of L, then use 
    :param Y: MCAR process observation, (d, N+1) np.array
    :param p: MCAR parameter, int
    :param P: finer partition over which we observe the MCAR process [0 = s_0, ..., s_N = t], (N+1,) np.array
    :param Q: coarser partition over which we recover the Levy increments [0 = u_0, ..., u_M = T], (M+1,) np.array
    :param AA: MCAR parameters, list of p (d, d) np.arrays
    :return DeltaL_Q: increments of the background driving Levy process L on the pratition Q, (d, M) np.array
    """
    # get dimensions and time horizon
    d = Y.shape[0]

    # approximate the state space representation by finite differences over P
    X_P = state_space(Y, p, P)
    Delta_P = np.diff(P)

    # find elements of Q in P
    indices = [False] * len(P)
    j = 0
    for i, s in enumerate(P):
        if s == Q[j]:
            indices[i] = True
            j += 1

    # restrict X to Q
    X_Q = X_P[:, indices]
    DeltaX_Q = np.diff(X_Q, axis = 1)

    # approximate drift on P
    drift_P = np.zeros((d, len(Delta_P)))
    for j in range(p):
        DjY = X_P[j*d:(j+1)*d, :-1]
        drift_P += AA[p-j-1].dot(DjY) * Delta_P.T

    # coarsen drift to Q
    cum_drift = np.cumsum(drift_P, axis=1)
    drift_Q = np.diff(np.concatenate([np.zeros((d, 1)), cum_drift[:, indices[:-1]]], axis=1), axis=1)

    # return L on Q
    DeltaL_Q = DeltaX_Q[-d:, :] + drift_Q

    return DeltaL_Q

def disentangle_BM(DeltaL: np.array, Q: np.array, Sigma: np.array, b: np.array, gamma: float = 1.01):
    """
    Disentangle Brownian-only increments on the partition Q.
    To define the critical region B_N we generalize the approach in (Gegler, 2011) to irregularly spaced data.
    :param DeltaL: increments of Levy process on the partition Q, (d, N) np.array
    :param Q: partition, [0 = u_0, ..., u_{N} = T], (N+1,) np.array
    :param Sigma: covariance of the Brownian part of the Levy process, (d, d) np.array
    :param b: drift of Levy process, (d,) np.array
    :param gamma: hyperparameter for defining the critical region >1, float
    :return DeltaW: subset of elements of DeltaL corresponding to the diffusion part only, (d, M) np.array with M <= N
            DeltaQ_W: time increments corresponding to the elements in DeltaW, (M,) np.array
    """
    # identify critical region x^T Sigma_inv x <= beta*2*Delta*log(N)
    DeltaQ = np.diff(Q)
    N = len(DeltaQ)
    subset = np.einsum('ji,ki->i', np.linalg.inv(Sigma).dot(DeltaL), DeltaL) < 2*gamma*DeltaQ*np.log(N)
    # return Brownian increments
    DeltaQ_W = DeltaQ[subset]
    DeltaW = DeltaL[:, subset] - np.outer(b, DeltaQ_W)
    return DeltaW, DeltaQ_W

def estimate_Sigma(DeltaW: np.array, DeltaQ: np.array):
    """
    Estimate the covariance matrix Sigma from the (irregularly spaced) Brownian increments DeltaW.
    :param DeltaW: Brownian increments, (d, M) 
    :param DeltaQ: time increments corresponding to the elements in DeltaW, (M,) np.array
    :param Sigma_hat: estimated covariance of the Brownian increments, (d, d) np.array
    """
    d = DeltaW.shape[0]
    if len(DeltaQ) == 0:
        return np.zeros((d,d))
    DeltaZ = DeltaW / np.sqrt(DeltaQ)
    Sigma_hat = np.einsum('ki,ji->kj', DeltaZ, DeltaZ) / DeltaZ.shape[1]
    return Sigma_hat

def estimate_Sigma_L(DeltaL: np.array, Q: np.array, b: np.array, epsilon: float = 0.01, gamma: float = 1.01, max_iter: int = 100):
    """
    Estimate the covariance matrix Sigma of the Brownian component from (irregularly spaced) Levy increments. 
    Use the iterative approach in (Gegler, 2011) Section 4.
    :param DeltaL: increments of Levy process on the partition Q, (d, N) np.array
    :param Q: partition, [0 = u_1, ..., u_N = T], (N+1,) np.array
    :param b: drift of Levy process, (d,) np.array
    :param epsilon: tolerance level for defining convergence of iterative scheme, float
    :param gamma: hyperparameter for defining the critical region >1, float
    :return Sigma_hat: estimated covariance of the Brownian component, (d, d) np.array
    """
    # initialize procedure
    DeltaQ_W = np.diff(Q)
    DeltaW = DeltaL - np.outer(b, DeltaQ_W)
    Sigma_hat_new = estimate_Sigma(DeltaW, DeltaQ_W)
    converged = False
    # keep track of Sigmas to see if enter a loop
    Sigmas = [Sigma_hat_new]
    while not converged:
        Sigma_hat_old = Sigma_hat_new
        DeltaW, DeltaQ_W = disentangle_BM(DeltaL, Q, Sigma_hat_old, b, gamma=gamma)
        Sigma_hat_new = estimate_Sigma(DeltaW, DeltaQ_W)
        converged = np.linalg.norm(Sigma_hat_new - Sigma_hat_old) <= epsilon

        # check if entering a loop
        if Sigma_hat_new in Sigmas:
            converged = True
            Sigma_hat_new = np.mean(Sigmas[Sigmas.index(Sigma_hat_new):], axis = 0)
        else:
            Sigmas.append(Sigma_hat_new)
    return Sigma_hat_new

def choose_nu(DeltaL: np.array, Q: np.array, b: np.array, epsilon: float = 0.01, gamma: float = 1.01):
    """
    This function chooses thresholding powers beta to disentangle the continuous component from the jump 
    component of a one dimenisonal Levy process.
    :param DeltaL: increments of Levy process on the partition Q, (1, N) np.array
    :param Q: partition, [0 = u_1, ..., u_N = T], (N+1,) np.array
    :param b: drift of Levy process, (1,) np.array
    :param epsilon: tolerance level for defining convergence of iterative scheme, float
    :param gamma: hyperparameter for defining the critical region >1, float
    :return nu: thresholding vector, (1, N) np.array
    """
    DeltaQ = np.diff(Q)
    Sigma_hat = estimate_Sigma_L(DeltaL, Q, b, epsilon, gamma)
    nu = np.sqrt(DeltaQ * 2 * gamma * np.log(len(DeltaQ)) * Sigma_hat)
    return nu