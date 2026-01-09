from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import cvxpy as cp


@dataclass(frozen=True)
class CVXICNN2LayerParams:
    """
    Container for CVXPY Parameters of a 2-layer ICNN-ish value function.

    Network (with skip connections from input s to each layer):
        y1 = W1 @ s + b1
        z1 = relu(y1)          (implemented with aux var + constraints + penalty)

        y2 = W2z @ z1 + W2x @ s + b2
        z2 = relu(y2)

        V  = Woutz @ z2 + Woutx @ s + bout

    Shapes:
        s: (input_dim, 1)
        z1: (h1, 1)
        z2: (h2, 1)
        V: (1, 1)
    """
    W1: cp.Parameter
    b1: cp.Parameter
    W2z: cp.Parameter
    W2x: cp.Parameter
    b2: cp.Parameter
    Woutz: cp.Parameter
    Woutx: cp.Parameter
    bout: cp.Parameter

    @property
    def parameter_list(self) -> List[cp.Parameter]:
        return [self.W1, self.b1, self.W2z, self.W2x, self.b2, self.Woutz, self.Woutx, self.bout]


class CVXICNN2Layer:
    """
    CVXPY ICNN building block that is DPP-friendly via auxiliary variables per ReLU layer.

    This is embedded into the policy CVXPY problem:
        - provide an input expression s (cvxpy Expression, shape (input_dim,1))
        - this class creates aux variables z1, z2 and returns:
            V_expr, constraints, penalty_expr, params
    """

    def __init__(
        self,
        input_dim: int,
        h1: int = 64,
        h2: int = 64,
        relu_penalty: float = 1.0,
        use_hard_relu_constraints: bool = True,
        name: str = "icnn",
    ):
        self.input_dim = int(input_dim)
        self.h1 = int(h1)
        self.h2 = int(h2)
        self.relu_penalty = float(relu_penalty)
        self.use_hard_relu_constraints = bool(use_hard_relu_constraints)
        self.name = str(name)

    def make_parameters(self) -> CVXICNN2LayerParams:
        """
        Create CVXPY Parameters for the network weights/biases.
        """
        W1 = cp.Parameter((self.h1, self.input_dim), name=f"{self.name}_W1")
        b1 = cp.Parameter((self.h1, 1), name=f"{self.name}_b1")

        # z-path weights (often constrained >=0 for ICNN property)
        W2z = cp.Parameter((self.h2, self.h1), name=f"{self.name}_W2z")
        # skip x-path weights
        W2x = cp.Parameter((self.h2, self.input_dim), name=f"{self.name}_W2x")
        b2 = cp.Parameter((self.h2, 1), name=f"{self.name}_b2")

        Woutz = cp.Parameter((1, self.h2), name=f"{self.name}_Woutz")
        Woutx = cp.Parameter((1, self.input_dim), name=f"{self.name}_Woutx")
        bout = cp.Parameter((1, 1), name=f"{self.name}_bout")

        return CVXICNN2LayerParams(W1=W1, b1=b1, W2z=W2z, W2x=W2x, b2=b2, Woutz=Woutz, Woutx=Woutx, bout=bout)

    def build(
        self,
        s: cp.Expression,
        params: Optional[CVXICNN2LayerParams] = None,
    ) -> Tuple[cp.Expression, List[cp.Constraint], cp.Expression, CVXICNN2LayerParams]:
        """
        Build the ICNN expressions/constraints.

        Args:
            s: CVXPY Expression, shape (input_dim,1). Example: s = vstack([x_next, x_ref])
            params: optionally pass pre-created parameters (so you can reuse across calls)

        Returns:
            V: CVXPY Expression scalar (1,1)
            constraints: list of CVXPY constraints for the ReLU aux vars
            penalty: CVXPY Expression scalar added to objective to "tie" z to y
            params: the parameter bundle
        """
        if params is None:
            params = self.make_parameters()

        # Validate shape if possible
        if s.shape != (self.input_dim, 1):
            raise ValueError(f"ICNN input s must have shape {(self.input_dim,1)}, got {s.shape}")

        # Auxiliary ReLU outputs (variables)
        z1 = cp.Variable((self.h1, 1), name=f"{self.name}_z1")
        z2 = cp.Variable((self.h2, 1), name=f"{self.name}_z2")

        # Pre-activations
        y1 = params.W1 @ s + params.b1
        y2 = params.W2z @ z1 + params.W2x @ s + params.b2

        constraints: List[cp.Constraint] = []
        penalty = 0

        # ReLU layer 1
        # Hard constraints for ReLU epigraph: z1 >= 0 and z1 >= y1
        constraints.append(z1 >= 0)
        if self.use_hard_relu_constraints:
            constraints.append(z1 >= y1)

        # Tie z1 to y1
        penalty += cp.sum_squares(z1 - y1)

        # ReLU layer 2
        constraints.append(z2 >= 0)
        if self.use_hard_relu_constraints:
            constraints.append(z2 >= y2)

        penalty += cp.sum_squares(z2 - y2)

        # Output layer
        V = params.Woutz @ z2 + params.Woutx @ s + params.bout  # (1,1)

        # Scale penalty
        penalty = self.relu_penalty * penalty

        return V, constraints, penalty, params