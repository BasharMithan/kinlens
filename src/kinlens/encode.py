"""
Module: Encoder: Responsible for encoding the target/label classes to numeric values,
so the model can understand it (classes = [A, B, C] -> {A: 1, B: 2, C: 3}).

Developer: Bashar Mithan

Date: 2026, 10, 26 
"""

import numpy as np


class Encoder:

    @staticmethod    
    def encodeLabels(labels: np.ndarray):
        """
        Convert categorical labels to integer codes.

        Returns:
            encoded: np.ndarray of integers
            labelToCode: dict mapping original label -> integer
            codeToLabel: dict mapping integer -> original label
        """

        uniqueLabels = np.unique(labels)

        labelToCode = {str(label): i for i, label in enumerate(uniqueLabels)}
        codeToLabel = {i: str(label) for label, i in labelToCode.items()}

        encoded = np.array([labelToCode[label] for label in labels], dtype=int)
        return encoded, labelToCode, codeToLabel