import numpy as np
from scipy.integrate import solve_ivp

def normalize(data: np.ndarray) -> np.ndarray:
    # FIX #3: use abs-max so negative-dominant signals (ground state bleach)
    # normalize to [-1, 1] instead of producing values far outside [0, 1].
    data_max = np.max(np.abs(data))
    if data_max == 0:
        return data
    return data / data_max

def gaussian(x: np.ndarray, mu: float, sig: float) -> np.ndarray:
    return 1. / (np.sqrt(2. * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.) / 2)

def data_dy(t: float, y_in: np.ndarray, kinetic_matrix: np.ndarray, irf_sigma: float = None) -> np.ndarray:
    y_out = np.dot(kinetic_matrix, y_in)
    
    if irf_sigma is not None:
        val = 0.1 * gaussian(np.array([t]), 0, irf_sigma)[0]
        y_out[0] += val
        y_out[-1] -= val
    return y_out

def solve_kinetics(
    t_eval: np.ndarray,
    K: np.ndarray,
    sigma_irf: float,
    interpolate: bool = True,
    tight_tol: bool = False,
) -> np.ndarray:
    """
    Solves ODE for concentration profiles.

    Args:
        t_eval:     Time points to evaluate at.
        K:          Kinetic rate matrix.
        sigma_irf:  IRF sigma in same units as t_eval.
        interpolate: If True return (len(t_eval), num_species) array.
                     If False return (step_c, step_t) for the adaptive solver steps.
        tight_tol:  If True use high-precision tolerances (1e-10) suitable for
                    real-data fitting.  If False (default) use relaxed tolerances
                    (1e-6) which are ~10x faster and sufficient for synthetic
                    training-data generation.

    Returns:
        Concentration profiles.
    """
    # FIX #2: relaxed tolerances for synthetic data generation (default).
    # tight_tol=True preserves the original 1e-10 accuracy for real-data fitting.
    atol = 1e-10 if tight_tol else 1e-6
    rtol = 1e-10 if tight_tol else 1e-6

    num_s = K.shape[0]
    c_0 = np.zeros(num_s)
    c_0[-1] = 1.0  # Ground state is fully populated initially

    t_start = -(10 * sigma_irf)
    t_end = t_eval[-1]

    def dy(t, y):
        return data_dy(t, y, K, sigma_irf)

    sol = solve_ivp(
        dy, (t_start, t_end), c_0,
        t_eval=t_eval if interpolate else None,
        method='RK45', atol=atol, rtol=rtol,
    )

    if interpolate:
        if sol.success:
            return sol.y.T
        else:
            # Fall back to dense output if solver struggled
            sol = solve_ivp(
                dy, (t_start, t_end), c_0,
                method='RK45', atol=atol, rtol=rtol, dense_output=True,
            )
            return sol.sol(t_eval).T
    else:
        return sol.y.T, sol.t

def convolve_irf(t: np.ndarray, k: float, u: float, D: float) -> np.ndarray:
    from scipy.special import erf
    return 0.5 * np.exp(-k * t) * np.exp(k * (u + ((k * D**2) / 2))) * (1 + erf((t - (u + k * D**2)) / (np.sqrt(2) * D)))
