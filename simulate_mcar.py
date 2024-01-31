import numpy as np
import scipy
import scipy.linalg
import scipy.stats

def reshape_array(array, d):
    if d == 1:
        return array.reshape(-1, 1)
    elif len(array.shape) == 1:
        return array.reshape(1, -1)
    else:
        return array

# Compute MCAR structural matrix from parameters AA
def MCAR_A(AA: np.array) -> np.array:
    """
    Construct MCAR structural matrix from parameters A.
    :param AA: MCAR parameters, list of p (d, d) np.arrays
    :return A_AA: MCAR structural matrix, (pd, pd) np.array
    """
    # get dimensions
    p = len(AA)
    d = AA[0].shape[0]
    # compute MCAR structural matrix
    A_AA = np.zeros([p*d, p*d])
    for i in range(d, p*d, d):
        A_AA[(i-d):i, i:(i+d)] = np.eye(d)
    for i in range(p):
        A_AA[-d:, i*d:(i+1)*d] = - AA[p-i-1]
    return A_AA

# Compute grCAR structural matrix from parameters theta
def grCAR_A(theta: np.array, A: np.array) -> np.array:
    """
    Construct grCAR structural matrix from parameters theta.
    :param theta: grCAR parameters, (p, 2) np.array
    :param A: graph adjacency matrix, (d, d) np.array
    :return A_theta: grCAR structural matrix, (pd, pd) np.array
    """
    # get dimensions
    p = theta.shape[0]
    d = A.shape[0]
    # compute normalised adjacency matrix
    barA = np.matmul(A, np.diag(1/np.sum(A, axis=0)))
    # compute MCAR coefficient matrices 
    AA = []
    for i in range(p):
        AA.append(theta[i, 0] * np.eye(d) + theta[i, 1] * barA.T)
    # compute grCAR structural matrix 
    A_theta = MCAR_A(AA)
    return A_theta

def simulate_MCAR_stat_distr_approx(A: np.array, b: np.array, Sigma: np.array, jumps):
    """
    Simulate from stationary distribution of MCAR process by approximating integral discretely.
    :param A: MCAR structural matrix, (pd, pd) np.array
    :param b: drift of Levy process, (d,) np.array
    :param Sigma: covariance of Brownian component of Levy process, (d, d) np.array
    :param jumps: function for generating n jump increments over an interval delta_t, function(delta_t, n)
    :return x: sample from stationary distribution, (d,) np.array
    """
    pd = A.shape[0]
    d = b.shape[0]
    E = np.zeros([pd, d])
    E[-d:,:] = np.eye(d)
    # approximate integral, choose T depending on |A| (depending on how fast e^As decays)
    # TODO: explore better ways to approximate this integral.
    T = - np.log(1e-8) / np.linalg.norm(A)
    N = 1000
    P = np.linspace(0, T, N+1)
    delta_t = T/N
    delta_jump_L = jumps(delta_t, N)
    delta_L = scipy.stats.multivariate_normal(mean=b*delta_t, cov=Sigma*delta_t).rvs(size=N).T + delta_jump_L
    return np.tensordot(scipy.linalg.expm(np.tensordot(P[:-1], A, axes=0)), np.matmul(E, delta_L), axes=[[0, 2], [1, 0]])
    
def simulate_MCAR_stat_distr_compound_poisson(A: np.array, b: np.array, Sigma: np.array, rate: float, jump_F: scipy.stats._multivariate):
    """
    Simulate from stationary distribution of finite activity MCAR process exactly.
    :param A: MCAR structural matrix, (pd, pd) np.array
    :param b: drift of Levy process, (d,) np.array
    :param Sigma: covariance of Brownian component of Levy process, (d, d) np.array
    :param jumps: function for generating n jump increments over an interval delta_t, function(delta_t, n)
    :return x: sample from stationary distribution, (d,) np.array
    """
    pd = A.shape[0]
    d = b.shape[0]
    E = np.zeros([pd, d])
    E[-d:,:] = np.eye(d)
    Sigma_tilde = E.dot(Sigma).dot(E.T)
    T = - np.log(1e-8) / np.linalg.norm(A)

    # precompute for speed-up
    M = np.zeros([2*pd, 2*pd])
    M[:pd, :pd] = A
    M[:pd, pd:] = Sigma_tilde
    M[pd:, pd:] = - A.T
    V = scipy.linalg.expm(M*T)[:pd, pd:].dot(scipy.linalg.expm(A*T).T)

    b_component = - np.linalg.inv(A).dot(E).dot(b)
    W_component = scipy.stats.multivariate_normal(cov=V, allow_singular=True).rvs(size=1).T

    # increment due to jumps
    N_T = scipy.stats.poisson(mu=rate*T).rvs(size=1)
    jump_times = np.sort(scipy.stats.uniform().rvs(size=N_T)*T)
    jump_sizes = reshape_array(jump_F.rvs(size=N_T), d).T
    J_component = np.einsum('ijk,ki->j',scipy.linalg.expm(np.tensordot(jump_times, A, axes=0)), E.dot(jump_sizes))

    x = b_component + W_component + J_component
    return x

def simulate_MCAR_approx(P: np.array, A: np.array, x0: np.array, b: np.array, Sigma: np.array, jumps, output_format: str = 'MCAR', uniform=False):
    """
    Simulate (discrete) paths from a MCAR model with structural matrix A and driving Levy process (with finite Levy measure)
    with triplet (b, Sigma, F) using Euler-Maruyama method:
    :param P: partition over which to simulate the MCAR process [0 = t_0, t_1, ..., t_N = T], (N+1,) np.array
    :param A: MCAR structural matrix, (pd, pd) np.array
    :param x0: state space initial condition, (pd,) np.array
    :param b: drift of Levy process, (d,) np.array
    :param Sigma: covariance of Brownian component of Levy process, (d, d) np.array
    :param jumps: function for generating n jump increments over an time interval delta_t, function(delta_t, n)
    :param output_format: format of output, str in ['MCAR', 'SS', 'SS + L', 'SS + L + jump_L']
    :return Y: MCAR simulation, (d, N+1) np.array
            X: MCAR state space simulation (first d entries are Y), (pd, N+1) np.array
            L: driving Levy process, (d, N+1) np.array
            jump_L: jumps in the driving Levy process, (d, N+1) np.array
    """
    # get dimensions and time step
    d = len(b)
    pd = len(x0)
    N = len(P)
    
    # compute E
    E = np.zeros([pd, d])
    E[-d:,:] = np.eye(d)
    
    # initialize 
    X = np.zeros([pd, N])
    L = np.zeros([d, N])
    jump_L = np.zeros([d, N])
    X[:, 0] = x0

    # simulate driving Brownian noise
    delta_W = np.sqrt(np.diff(P))[np.newaxis, :] * scipy.stats.multivariate_normal(cov=Sigma).rvs(size=N-1).T
    delta_L = np.zeros((d, N-1))

    if uniform:
        delta_t = P[1] - P[0]
        delta_jump_L = jumps(delta_t, N-1)
        delta_L = b.reshape(-1, 1) * delta_t + delta_W + delta_jump_L
        jump_L[:, 1:] = delta_jump_L.cumsum(axis=1)

        # get Levy process
        L[:, 1:] = delta_L.cumsum(axis=1)
    else: 
        for n in range(N-1):
            delta_t = P[n+1] - P[n]
            
            # Levy increment (continuous and jump parts)
            delta_jump_L = jumps(delta_t, 1).reshape(-1, 1)
            delta_L[:, n] = b.reshape(-1, 1) * delta_t + delta_W[:, n] + delta_jump_L
            
            # evolve the processes
            jump_L[:, n + 1] = jump_L[:, n] + delta_jump_L
            L[:, n + 1] = L[:, n] + delta_L[:, n]
    
    # evolve state space
    for n in range(N-1):
        X[:, n+1] = X[:, n] + np.matmul(A, X[:, n]) * delta_t + np.matmul(E, delta_L[:, n])

    if output_format == 'MCAR':
        return X[:d, :]
    elif output_format == 'SS':
        return X
    elif output_format == 'SS + L':
        return X, L
    elif output_format == 'SS + L + jump_L':
        return X, L, jump_L
    else:
        raise ValueError("output_format must be one of ['MCAR', 'SS', 'SS + L', 'SS + L + jump_L']")

def simulate_MCAR_compound_poisson(P: np.array, A: np.array, x0: np.array, b: np.array, Sigma: np.array, rate: float, jump_F: scipy.stats._multivariate, output_format: str = 'MCAR', uniform=False):
    """
    Simulate (discrete) paths from a MCAR model with structural matrix A and driving Levy process (with finite Levy measure)
    with triplet (b, Sigma, F).
    X_t = e^{A(t-s)} X_s + int_s^t e^{A(t-r)} dL_r = e^{A(t-s)}X_s + int_s^t e^{A(t-r)} (b dr + dW_r + dJ_r)
    where int_s^t e^{A(t-r)} b dr = (e^{A(t-s)} - I) A^{-1} b
          int_s^t e^{A(t-r)} dW_r ~ N(0, F(t-s) e^{A^T(t-s)}) where e^Mh = [e^{Ah}, F(h) // 0, e^{-A^Th}] with M = [A, Σ // 0, -A^T] for W~N(0, Σ)
          int_s^t e^{A(t-r)} dJ_r = sum_{r\in[s,t]} e^{A(t-r)} ΔJ_{r}
    recall need to multiply E b, E Sigma E^T, and E jumps
    :param P: partition over which to simulate the MCAR process [0 = t_0, t_1, ..., t_N = T], (N+1,) np.array
    :param A: MCAR structural matrix, (pd, pd) np.array
    :param x0: state space initial condition, (pd,) np.array
    :param b: drift of Levy process, (d,) np.array
    :param Sigma: covariance of Brownian component of Levy process, (d, d) np.array
    :param jumps: function for generating n jump component increments over an interval delta_t, function(delta_t, n)
    :param output_format: format of output, str in ['MCAR', 'SS']
    :return if output_format = 'MCAR':
                Y: MCAR simulation, (d, N+1) np.array
            if output_format = 'SS':
                X: MCAR state space simulation (first d entries are Y), (pd, N+1) np.array
    """
    # get dimensions and time step
    d = len(b)
    pd = len(x0)
    N = len(P)
    T = P[-1] - P[0]
    
    # compute E, Sigma_tilde
    E = np.zeros([pd, d])
    E[-d:,:] = np.eye(d)
    Sigma_tilde = E.dot(Sigma).dot(E.T)
    
    # initialize 
    X = np.zeros([pd, N])
    X[:, 0] = x0

    # precompute for speed-up
    eA = scipy.linalg.expm(A)
    A_inv = np.linalg.inv(A)
    M = np.zeros([2*pd, 2*pd])
    M[:pd, :pd] = A
    M[:pd, pd:] = Sigma_tilde
    M[pd:, pd:] = - A.T
    eM = scipy.linalg.expm(M)
    
    if uniform:
        delta_t = P[1] - P[0]
        eAt = scipy.linalg.fractional_matrix_power(eA, delta_t).real
        
        # increment due to b
        b_increment = (eAt - np.eye(pd)).dot(A_inv).dot(E).dot(b)
        
        # increment due to W
        V = scipy.linalg.fractional_matrix_power(eM, delta_t).real[:pd, pd:].dot(eAt.T)
        W_increments = scipy.stats.multivariate_normal(cov=V, allow_singular=True).rvs(size=N-1).T

        # increment due to jumps
        J_increments = np.zeros([pd, N-1])
        N_T = scipy.stats.poisson(mu=rate*T).rvs(size=1)
        jump_times = np.sort(scipy.stats.uniform().rvs(size=N_T)*T)
        jump_sizes = reshape_array(jump_F.rvs(size=N_T), d).T
        jump_indices = (jump_times // delta_t).astype(int)
        for i, jump_index in enumerate(jump_indices):
            J_increments[:, jump_index] += scipy.linalg.fractional_matrix_power(eA, (jump_index+1)*delta_t - jump_times[i]).real.dot(E).dot(jump_sizes[:, i])
        
        # evolve the process
        for n in range(N-1):
            X[:, n+1] = eAt.dot(X[:, n]) + b_increment + W_increments[:, n] + J_increments[:, n]
    else: 
        for n in range(N-1):
            delta_t = P[n+1] - P[n]
            eAt = scipy.linalg.fractional_matrix_power(eA, delta_t).real

            # increment due to b
            b_increment = (eAt - np.eye(pd)).dot(A_inv).dot(E).dot(b)
            
            # increment due to W
            M = np.zeros([2*pd, 2*pd])
            M[:pd, :pd] = A
            M[:pd, pd:] = Sigma_tilde
            M[pd:, pd:] = - A.T
            V = scipy.linalg.fractional_matrix_power(eM, delta_t).real[:pd, pd:].dot(eAt.T)
            W_increment = scipy.stats.multivariate_normal(cov=V, allow_singular=True).rvs(size=1).T

            # increment due to jumps
            N_delta = int(scipy.stats.poisson(mu=rate*delta_t).rvs(size=1))
            jump_sizes = reshape_array(jump_F.rvs(size=N_delta), d)
            jump_times = np.sort(scipy.stats.uniform().rvs(size=N_delta)*delta_t)
            J_increment = 0
            for _ in range(N_delta):
                J_increment += scipy.linalg.fractional_matrix_power(eA, delta_t - jump_times[_]).real.dot(E).dot(jump_sizes[:, _])

            # evolve the process
            X[:, n+1] = eAt.dot(X[:, n]) + b_increment + W_increment + J_increment

    if output_format == 'MCAR':
        return X[:d, :]
    elif output_format == 'SS':
        return X
    else:
        raise ValueError("output_format must be one of ['MCAR', 'SS']")
    

def compound_poisson(delta_t: float, n: int, rate: float, jump_F: scipy.stats._multivariate, d: int):
    """
    Generate n increments of a compound Poisson process each over an interval of size delta_t
    :param delta_t: time step of increment, float
    :param n: number of increments to generate, int
    :param rate: jump rate of Levy process, int
    :param jump_F: distribution of jumps, multivariate scipy.stats distribution
    :param d: dimension of process, int
    :return increments: the increments of the process, (d, n) np.array
    """
    T = n*delta_t
    increments = np.zeros([d, n])
    N_T = scipy.stats.poisson(mu=rate*T).rvs(size=1)
    jump_times = np.sort(scipy.stats.uniform().rvs(size=N_T)*T)
    jump_sizes = reshape_array(jump_F.rvs(size=N_T), d).T
    jump_indices = (jump_times // delta_t).astype(int)
    for i, jump_index in enumerate(jump_indices):
        increments[:, jump_index] += jump_sizes[:, i]
    return increments

def gamma_increments(delta_t: float, n: int, d: int, shape: float, scale: float = 1):
    """
    Generate n increments of a d-dimenisonal process with independent symmetric Gamma components
    over an interval of size delta_t.
    :param delta_t: time step of increment, float
    :param n: number of increments to generate, int
    :param shape: shape of Gamma distribution, float
    :param scale: scale of Gamma distribution, float
    :param d: dimension of process, int
    :return increments: the increments of the process, (d, n) np.array
    """
    increments = (scipy.stats.gamma.rvs(a=shape*delta_t, scale=scale, size=d*n) - scipy.stats.gamma.rvs(a=delta_t, size=d*n)).reshape((d, -1))
    return increments