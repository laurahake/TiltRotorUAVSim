import numpy as np


class ReplayBuffer:
    """
    State-Replay Buffer for augmented MDP samples.

    Stores tuples:
        (x, xref, theta, done, reason)

    Notes:
    - encodes 'reason' as int for efficiency by default.
    - sample() returns numpy arrays for easy batching.
    """

    REASON_TO_ID = {
        "none": 0,
        "pos": 1,
        "vel": 2,
        "angle": 3,
        "omega": 4,
        "tilt": 5,
        "quat_norm": 6,
        "other": 7,
    }
    ID_TO_REASON = {v: k for k, v in REASON_TO_ID.items()}

    def __init__(self, max_size: int, nx: int = 15, nu: int = 5, theta_dim: int = 2, store_reason_as_int: bool = True):
        self.mem_size = int(max_size)
        self.nx = int(nx)
        self.nu = int(nu)
        self.theta_dim = int(theta_dim)
        self.store_reason_as_int = bool(store_reason_as_int)

        self.mem_counter = 0  # total number of writes (can exceed mem_size)

        # Preallocate as numpy arrays (faster and simpler than Python lists)
        self.x_memory = np.zeros((self.mem_size, self.nx), dtype=np.float32)
        self.xref_memory = np.zeros((self.mem_size, self.nx), dtype=np.float32)
        self.theta_memory = np.zeros((self.mem_size, self.theta_dim), dtype=np.float32)
        self.u_prev_memory = np.zeros((self.mem_size, self.nu), dtype=np.float32)

        self.done_memory = np.zeros((self.mem_size,), dtype=np.bool_)

        if self.store_reason_as_int:
            self.reason_memory = np.zeros((self.mem_size,), dtype=np.int32)
        else:
            self.reason_memory = np.empty((self.mem_size,), dtype=object)

    def __len__(self):
        return min(self.mem_counter, self.mem_size)

    def is_ready(self, min_size: int) -> bool:
        return len(self) >= int(min_size)

    @staticmethod
    def _as_float32_1d(x, expected_dim: int, name: str) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32).reshape(-1)
        if arr.size != expected_dim:
            raise ValueError(f"{name} must have dim {expected_dim}, got {arr.size}")
        return arr

    def _encode_reason(self, reason):
        if reason is None:
            return 0 if self.store_reason_as_int else "none"

        if self.store_reason_as_int:
            if isinstance(reason, (int, np.integer)):
                return int(reason)
            reason_str = str(reason)
            return int(self.REASON_TO_ID.get(reason_str, self.REASON_TO_ID["other"]))
        else:
            return str(reason)

    def store(self, x, xref, theta, u_prev, done: bool, reason=None):
        """
        Store one sample (x, xref, theta, done, reason).
        """
        idx = self.mem_counter % self.mem_size

        x_arr = self._as_float32_1d(x, self.nx, "x")
        xref_arr = self._as_float32_1d(xref, self.nx, "xref")
        theta_arr = self._as_float32_1d(theta, self.theta_dim, "theta")
        u_prev_arr = self._as_float32_1d(u_prev, self.nu, "u_prev")
        
        self.u_prev_memory[idx] = u_prev_arr
        self.x_memory[idx] = x_arr
        self.xref_memory[idx] = xref_arr
        self.theta_memory[idx] = theta_arr

        self.done_memory[idx] = bool(done)
        self.reason_memory[idx] = self._encode_reason(reason)

        self.mem_counter += 1

    def sample(self, batch_size: int, replace: bool = False, rng: np.random.Generator = None):
        """
        Sample a batch of samples.

        Returns:
            x     : (B, nx) float32
            xref  : (B, nx) float32
            theta : (B, 2)  float32
            done  : (B,)    bool
            reason: (B,)    int32 or object
        """
        max_mem = len(self)
        if max_mem == 0:
            raise RuntimeError("ReplayBuffer is empty. Fill it before sampling.")

        batch_size = int(batch_size)
        if (not replace) and batch_size > max_mem:
            raise ValueError(f"batch_size={batch_size} > available={max_mem} with replace=False")

        rng = np.random.default_rng() if rng is None else rng
        idxs = rng.choice(max_mem, size=batch_size, replace=bool(replace))

        return (
            self.x_memory[idxs],
            self.xref_memory[idxs],
            self.theta_memory[idxs],
            self.u_prev_memory[idxs],
            self.done_memory[idxs],
            self.reason_memory[idxs],
        )

    # ----------------------------
    # Optional backward-compatible aliases
    # ----------------------------
    def store_transition(self, state, action, cost, next_state, terminate):
        """
        Backward-compat shim (optional).
        Interprets:
            state     -> x
            next_state-> xref
            action    -> theta
            terminate -> done
            cost      -> ignored (kept for call-site compatibility)
        """
        self.store(x=state, xref=next_state, theta=action, done=terminate, reason=None)

    def sample_buffer(self, batchsize):
        """
        Backward-compat shim (optional).
        Returns in the old order but mapped:
            states     -> x
            actions    -> theta
            costs      -> dummy zeros
            next_states-> xref
            terminates -> done
        """
        x, xref, theta, done, reason = self.sample(batchsize, replace=False)
        costs = np.zeros((len(done),), dtype=np.float32)
        return list(x), list(theta), list(costs), list(xref), list(done)
