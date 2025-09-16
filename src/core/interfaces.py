# src/core/interfaces.py

from abc import ABC, abstractmethod
from typing import List

class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """
        Nhận vào một chuỗi văn bản, trả về danh sách các token.
        """
        pass


class Vectorizer(ABC):
    """
    Interface cho Vectorizer.
    - fit: học vocabulary từ corpus
    - transform: biến đổi văn bản thành vector đếm
    - fit_transform: tiện ích kết hợp fit + transform
    """

    @abstractmethod
    def fit(self, corpus: List[str]) -> None:
        """
        Học vocabulary từ danh sách tài liệu (corpus).
        """
        pass

    @abstractmethod
    def transform(self, documents: List[str]) -> List[List[int]]:
        """
        Biến đổi danh sách tài liệu thành ma trận vector đếm,
        dựa trên vocabulary đã học.
        """
        pass

    @abstractmethod
    def fit_transform(self, corpus: List[str]) -> List[List[int]]:
        """
        Thực hiện fit và transform trên cùng một corpus.
        """
        pass
