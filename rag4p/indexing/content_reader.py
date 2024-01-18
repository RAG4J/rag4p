from abc import ABC, abstractmethod


class ContentReader(ABC):
    @abstractmethod
    def read(self):
        pass
