from typing import Optional, List, Tuple
import torch
from torch import Tensor
from botorch.test_functions.synthetic import SyntheticTestFunction
from discrete_mixed_bo.problems.base import DiscretizedBotorchTestProblem


class OneMax(SyntheticTestFunction):
    r"""OneMax test function.

    The goal is to maximize the sum of the bits in a binary string.

    For a binary vector x of length d:

        f(x) = sum_{i=1}^d x_i

    The global maximum is achieved when all bits are set to 1.
    """
    def __init__(
            self,
            dim: int = 10,
            noise_std: Optional[float] = None,
            negate: bool = False,
    ) -> None:
        r"""
        Args:
            dim: The dimension (length) of the binary vector.
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
        """
        self.dim = dim
        self._optimal_value = self.dim  # The optimal value for OneMax is the dimension
        self._optimizers = [tuple(1 for _ in range(self.dim))]  # The optimal solution is all 1s

        # Bounds for each dimension in the binary space (0 or 1)
        bounds = [(0, 1) for _ in range(self.dim)]

        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds)


    def evaluate_true(self, X: Tensor) -> Tensor:
        """
        Evaluate the OneMax function on a binary input X.

        Args:
            X: A binary tensor of shape (batch_size, dim).

        Returns:
            A tensor of shape (batch_size,) containing the function values.
        """
        # Calculate the sum of ones for each binary vector in the batch
        result = torch.sum(X, dim=-1)

        return result