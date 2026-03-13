import json
import numpy as np


class CVDMatrix:
    def __init__(self, model: int, type_cdo: int, degree: float,
                 matrix_file: str = "cvd_matrices.json"):
        self.model = int(model)
        self.type_cdo = int(type_cdo)
        self.degree = float(degree)
        self.matrix_file = matrix_file

    def _degree_key(self) -> str:
        d = round(self.degree * 10) / 10
        d = min(max(d, 0.0), 1.0)
        return f"{d:.1f}"

    def compute(self) -> np.ndarray:
        with open(self.matrix_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        model_key = str(self.model)
        type_key = str(self.type_cdo)
        degree_key = self._degree_key()

        try:
            matrix = data[model_key][type_key][degree_key]
        except KeyError:
            raise ValueError(
                f"Matrix not found for model={self.model}, "
                f"type_cdo={self.type_cdo}, degree={degree_key}"
            )

        return np.array(matrix, dtype=float)