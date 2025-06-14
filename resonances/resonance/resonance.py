from abc import ABC, abstractmethod


class Resonance(ABC):
    @abstractmethod
    def to_s(self):
        pass

    @abstractmethod
    def to_short(self):
        pass
