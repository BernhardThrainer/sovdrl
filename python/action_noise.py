from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

class ActionNoise(ABC):
    """The action noise base class"""
    def __init__(self):
        super().__init__()
    
    def reset(self) -> None:
        pass

    @abstractmethod
    def __call__(self) -> np.ndarray:
        raise NotImplementedError()
    
class OrnsteinUhlenbeckActionNoise(ActionNoise):
    def __init(
            self,
            mean: np.ndarray,
            sigma: np.ndarray,
            theta: float = 0.15,
            dt: float = 1e-2,
            seed: Optional[float] = None,
            initial_noise: Optional[np.ndarray] = None):
        self._theta = theta
        self._mu = mean
        self._sigma = sigma
        self._dt = dt
        self._seed = seed
        self.initial_noise = initial_noise
        self.noise_prev = np.zeros_like(self._mu)
        self.reset()
        np.random.seed(self._seed)
        super().__init__()
    
    def __call__(self) -> np.ndarray:
        noise = (
            self.noise_prev
            + self._theta * (self._mu - self.noise_prev) * self._dt
            + self._sigma * np.sqrt(self._dt) * np.random.normal(size = self._mu.shape)
        )
        self.noise_prev = noise
        return noise
    
    def reset(self) -> None:
        self.noise_prev = self.initial_noise if self.initial_noise is not None else np.zeros_like(self._mu)
    
    def __repr__(self) -> str:
        return f"OrnsteinUhlenbeckActionNoise(mu={self._mu}, sigma={self._sigma}) = {self.noise_prev}"