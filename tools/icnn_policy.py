import cvxpy as cp
import numpy as np

class ICNNPolicy:
    def __init__(self, nx, nu, hidden=64):
        self.nx = nx
        self.nu = nu
        self.hidden = hidden
        
        # build the CVXPY problem
        prob, u_var, params = self.build_cvx_layer(nx, nu, hidden)
        self.prob = prob
        self.u_var = u_var
        
        #unpack parameters
        (self.xk_p,
         self.xref_p,
         self.Ad_p,
         self.Bd_p,
         self.cd_p,
         self.Q_half_p,
         self.R_half_p,
         self.gamma_p,
         self.Wx1_p,
         self.b1_p,
         self.Az2_p,
         self.Wx2_p,
         self.b2_p,
         self.wout_p,
         self.bout_p,
         self.umin_p,
         self.umax_p) = params
        
        # Initialize ICNN weights ψ as NumPy arrays
        self.Wx1 = np.random.randn(hidden, nx) * 0.1
        self.b1  = np.zeros(hidden)
        self.Az2 = np.abs(np.random.randn(hidden, hidden)) * 0.1  # nonneg
        self.Wx2 = np.random.randn(hidden, nx) * 0.1
        self.b2  = np.zeros(hidden)
        self.wout = np.abs(np.random.randn(hidden)) * 0.1         # nonneg
        self.bout = np.array(0.0)
        

    def build_cvx_layer(self, nx, nu, hidden):
        # Variables & Parameters
        u   = cp.Variable(nu)
        xk  = cp.Parameter(nx)
        xref= cp.Parameter(nx)
        Ad  = cp.Parameter((nx, nx))
        Bd  = cp.Parameter((nx, nu))
        cd  = cp.Parameter(nx)
        Q_half = cp.Parameter((3, 3))   
        R_half = cp.Parameter((nu, nu))
        gamma = cp.Parameter(nonneg=True)

        # Constants not Parameters for DPP compliance
        Wx1 = cp.Parameter((hidden, nx))                    # free
        b1  = cp.Parameter(hidden)
        Az2 = cp.Parameter((hidden, hidden), nonneg=True)   # <-- MUST be nonnegative
        Wx2 = cp.Parameter((hidden, nx))                    # free
        b2  = cp.Parameter(hidden)
        wout= cp.Parameter(hidden, nonneg=True)             # <-- nonnegative for convex sum
        bout= cp.Parameter()                  

        # Next state (affine in u)
        xnext = Ad @ xk + Bd @ u + cd

        e_pos = xk[:3] - xref[:3]
        stage = cp.sum_squares(Q_half @ e_pos) + cp.sum_squares(R_half @ u)

        # Two-layer ICNN Ṽ(xnext)
        z1 = cp.pos(Wx1 @ xnext + b1)                 # ReLU
        z2 = cp.pos(Az2 @ z1 + Wx2 @ xnext + b2)      # ReLU, Az2 must be >= 0
        Vtilde = wout @ z2 + bout                     # linear, convex composition preserved

        # Box constraints on u
        umin = cp.Parameter(nu)
        umax = cp.Parameter(nu)
        cons = [u >= umin, u <= umax]

        obj = cp.Minimize(stage + gamma * Vtilde)
        prob = cp.Problem(obj, cons)
        

        # Expose in a dict
        params = [
        xk, xref, Ad, Bd, cd,
        Q_half, R_half,      
        gamma,
        Wx1, b1, Az2, Wx2, b2, wout, bout,
        umin, umax
    ]
        return prob, u, params
    
    
    # ---- helpers ----
    def get_params(self):
        """Flatten ψ into a 1D vector for SPSA."""
        parts = [
            self.Wx1.ravel(),
            self.b1.ravel(),
            self.Az2.ravel(),
            self.Wx2.ravel(),
            self.b2.ravel(),
            self.wout.ravel(),
            np.array([self.bout]).ravel(),
        ]
        return np.concatenate(parts)
    
    
    def set_params(self, psi_vec):
        """Set ψ from a 1D vector (inverse of get_params)."""
        h, nx = self.hidden, self.nx
        idx = 0

        def take(shape):
            nonlocal idx
            n = np.prod(shape)
            block = psi_vec[idx:idx+n].reshape(shape)
            idx += n
            return block

        self.Wx1 = take((h, nx))
        self.b1  = take((h,))
        self.Az2 = np.maximum(take((h, h)), 0.0)   # enforce nonneg
        self.Wx2 = take((h, nx))
        self.b2  = take((h,))
        self.wout = np.maximum(take((h,)), 0.0)    # enforce nonneg
        self.bout = take((1,))
        
        self.Wx1  = np.clip(self.Wx1,  -3, +3)
        self.Wx2  = np.clip(self.Wx2,  -3, +3)
        self.b1   = np.clip(self.b1,   -3, +3)
        self.b2   = np.clip(self.b2,   -3, +3)
        self.bout = float(np.clip(self.bout, -3, +3))


    def __call__(self, xk, xref, Ad, Bd, cd, Q, R, gamma, umin, umax):
        """
        Solve the COCP:
        u* = argmin_u stage(xk,u) + gamma * Vtilde(x_next; ψ)
        with current ψ.
        All args are NumPy arrays.
        """
        # Set dynamic parameters
        self.xk_p.value   = xk
        self.xref_p.value = xref
        self.Ad_p.value   = Ad
        self.Bd_p.value   = Bd
        self.cd_p.value   = cd

        # cost weights as Q_half, R_half
        self.Q_half_p.value = np.linalg.cholesky(Q)
        self.R_half_p.value = np.linalg.cholesky(R)

        self.gamma_p.value = float(gamma)

        # Set ICNN parameters
        self.Wx1_p.value = self.Wx1
        self.b1_p.value  = self.b1
        self.Az2_p.value = self.Az2
        self.Wx2_p.value = self.Wx2
        self.b2_p.value  = self.b2
        self.wout_p.value= self.wout
        self.bout_p.value= float(self.bout)

        # Input bounds
        self.umin_p.value = umin
        self.umax_p.value = umax

        # Solve convex problem
        self.prob.solve(solver=cp.SCS, verbose=False)

        if (self.u_var.value is None or
            self.prob.status not in ["optimal", "optimal_inaccurate"]):
            print("[ICNNPolicy] WARNING: solve failed with status",
                  self.prob.status)

            # Fallback
            u_fallback = 0.5 * (umin + umax)
            return u_fallback.astype(np.float32).flatten()

        u_star = np.asarray(self.u_var.value).astype(np.float32).flatten()
        return u_star

