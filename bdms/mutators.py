r"""Mutation effects generators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Abstract base class for defining generic mutation effect generators (i.e.
:math:`\mathcal{p}(x\mid x')`), with arbitrary :py:class:`ete3.TreeNode` attribute
dependence. Some concrete child classes are included.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Optional
from collections.abc import Iterable, Callable
import numpy as np
from scipy.stats import norm, gaussian_kde
import ete3

# NOTE: sphinx is currently unable to present this in condensed form when the
#       sphinx_autodoc_typehints extension is enabled
from numpy.typing import ArrayLike


class Mutator(ABC):
    r"""Abstract base class for mutators that mutate a specified
    :py:class:`ete3.TreeNode` node attribute.

    Args:
        attr: Node attribute to mutate.
    """

    def __init__(self, attr: str = "state", *args: Any, **kwargs: Any) -> None:
        self.attr = attr

    @abstractmethod
    def mutate(
        self,
        node: ete3.TreeNode,
        seed: Optional[int | np.random.Generator] = None,
    ) -> None:
        r"""Mutate a :py:class:`ete3.TreeNode` object in place.

        Args:
            node: A :py:class:`ete3.TreeNode` to mutate.
            seed: A seed to initialize the random number generation. If ``None``, then
                  fresh, unpredictable entropy will be pulled from the OS. If an
                  ``int``, then it will be used to derive the initial state. If a
                  :py:class:`numpy.random.Generator`, then it will be used directly.
        """

    def __repr__(self) -> str:
        keyval_strs = (
            f"{key}={value}"
            for key, value in vars(self).items()
            if not key.startswith("_")
        )
        return f"{self.__class__.__name__}({', '.join(keyval_strs)})"


class GaussianMutator(Mutator):
    r"""Gaussian mutation effects on a specified attribute.

    Args:
        shift: Mean shift wrt current attribute value.
        scale: Standard deviation of mutation effect.
        attr: Node attribute to mutate.
    """

    def __init__(
        self,
        shift: float = 0.0,
        scale: float = 1.0,
        attr: str = "state",
    ):
        super().__init__(attr=attr)
        self.shift = shift
        self.scale = scale
        self._distribution = norm(loc=self.shift, scale=self.scale)

    def mutate(
        self,
        node: ete3.TreeNode,
        seed: Optional[int | np.random.Generator] = None,
    ) -> None:
        new_value = getattr(node, self.attr) + self._distribution.rvs(random_state=seed)
        setattr(node, self.attr, new_value)

    def prob(self, attr1: float, attr2: float, log: bool = False) -> float:
        Δx = np.asarray(attr2) - np.asarray(attr1)
        return self._distribution.logpdf(Δx) if log else self._distribution.pdf(Δx)


class KdeMutator(Mutator):
    r"""Gaussian kernel density estimator (KDE) for mutation effect on a specified
    attribute.

    Args:
        dataset: Data to fit the KDE to.
        attr: Node attribute to mutate.
        bw_method: KDE bandwidth (see :py:class:`scipy.stats.gaussian_kde`).
        weights: Weights of data points (see :py:class:`scipy.stats.gaussian_kde`).
    """

    def __init__(
        self,
        dataset: ArrayLike,
        attr: str = "state",
        bw_method: Optional[str | float, Callable] = None,
        weights: Optional[ArrayLike] = None,
    ):
        super().__init__(attr=attr)
        self._distribution = gaussian_kde(dataset, bw_method, weights)

    def mutate(
        self,
        node: ete3.TreeNode,
        seed: Optional[int | np.random.Generator] = None,
    ) -> None:
        new_value = (
            getattr(node, self.attr)
            + self._distribution.resample(size=1, seed=seed)[0, 0]
        )
        setattr(node, self.attr, new_value)

    def prob(self, attr1: float, attr2: float, log: bool = False) -> float:
        Δx = np.asarray(attr2) - np.asarray(attr1)
        return self._distribution.logpdf(Δx) if log else self._distribution.pdf(Δx)


class DiscreteMutator(Mutator):
    r"""Mutations on a discrete space with a stochastic matrix.

    Args:
        state_space: hashable state values.
        transition_matrix: Right-stochastic matrix, where column and row orders match
                           the order of `state_space`.
        attr: Node attribute to mutate.
    """

    def __init__(
        self,
        state_space: Iterable,
        transition_matrix: ArrayLike,
        attr: str = "state",
    ):
        transition_matrix = np.asarray(transition_matrix, dtype=float)
        if not (
            transition_matrix.ndim == 2
            and transition_matrix.shape[0]
            == transition_matrix.shape[1]
            == len(state_space)
        ):
            raise ValueError(
                f"Transition matrix shape {transition_matrix.shape} "
                f"does not match state space size {state_space}"
            )
        if np.any(transition_matrix < 0) or np.any(
            np.abs(transition_matrix.sum(axis=1) - 1) > 1e-4
        ):
            raise ValueError("Transition matrix is not a valid stochastic matrix.")

        super().__init__(attr=attr)

        self.state_space = {state: index for index, state in enumerate(state_space)}
        self.transition_matrix = transition_matrix

    def mutate(
        self,
        node: ete3.TreeNode,
        seed: Optional[int | np.random.Generator] = None,
    ) -> None:
        rng = np.random.default_rng(seed)

        states = list(self.state_space.keys())
        transition_probs = self.transition_matrix[
            self.state_space[getattr(node, self.attr)], :
        ]
        new_value = rng.choice(states, p=transition_probs)
        setattr(node, self.attr, new_value)
