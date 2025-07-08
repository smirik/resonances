from abc import ABC, abstractmethod


class Resonance(ABC):
    @property
    @abstractmethod
    def type(self) -> str:
        pass

    @abstractmethod
    def to_s(self):
        pass

    @abstractmethod
    def to_short(self):
        pass
