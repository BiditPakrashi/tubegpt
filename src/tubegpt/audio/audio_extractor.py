from abc import ABC, abstractmethod


class AudioExtractor(ABC):

    @abstractmethod
    def audio_to_text(self,urls,save_dir):
        pass
