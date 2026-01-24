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

    def __str__(self):
        return f"{self.type}({self.to_short()})"

    def __repr__(self):
        return str(self)
