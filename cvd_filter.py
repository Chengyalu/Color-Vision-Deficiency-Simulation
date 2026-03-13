import numpy as np
import pandas as pd
from PIL import Image


class CVDFilter:
    def __init__(self, matrix: np.ndarray, image_path: str, model: int,
                 excel_path: str = "covmartixlmsrgb.xlsx"):
        self.COVMatrix = np.asarray(matrix, dtype=float)
        self.path = image_path
        self.model = model
        self.excel_path = excel_path

        self._pre_cal_ill()

    def _read(self):
        df = pd.read_excel(self.excel_path, header=None)
        self.R = df.iloc[:401, 4].to_numpy(dtype=float)
        self.G = df.iloc[:401, 5].to_numpy(dtype=float)
        self.B = df.iloc[:401, 6].to_numpy(dtype=float)

    def _pre_cal_ill(self):
        self._read()
        IL = np.array([self.R.sum(), self.G.sum(), self.B.sum()], dtype=float)

        self.redIL = np.zeros(256, dtype=float)
        self.greenIL = np.zeros(256, dtype=float)
        self.blueIL = np.zeros(256, dtype=float)

        self.redIL[255] = IL[0]
        self.blueIL[255] = IL[1]
        self.greenIL[255] = IL[2]

        for j in range(255):
            self.redIL[j] = (j ** 2.2) / (255 ** 2.2) * IL[0]
            self.blueIL[j] = (j ** 2.2) / (255 ** 2.2) * IL[1]
            self.greenIL[j] = (j ** 2.2) / (255 ** 2.2) * IL[2]

    @staticmethod
    def _clip255(x):
        if x < 0:
            return 0
        if x > 255:
            return 255
        return int(round(x))

    def compute_rgb(self, r: int, g: int, b: int):
        nr = self.COVMatrix[0, 0] * r + self.COVMatrix[0, 1] * g + self.COVMatrix[0, 2] * b
        ng = self.COVMatrix[1, 0] * r + self.COVMatrix[1, 1] * g + self.COVMatrix[1, 2] * b
        nb = self.COVMatrix[2, 0] * r + self.COVMatrix[2, 1] * g + self.COVMatrix[2, 2] * b
        return [self._clip255(nr), self._clip255(ng), self._clip255(nb)]

    def compute_rgb_new(self, r: int, g: int, b: int):
        dr = (r / 255.0) ** 2.2
        dg = (g / 255.0) ** 2.2
        db = (b / 255.0) ** 2.2

        dnr = self.COVMatrix[0, 0] * dr + self.COVMatrix[0, 1] * dg + self.COVMatrix[0, 2] * db
        dng = self.COVMatrix[1, 0] * dr + self.COVMatrix[1, 1] * dg + self.COVMatrix[1, 2] * db
        dnb = self.COVMatrix[2, 0] * dr + self.COVMatrix[2, 1] * dg + self.COVMatrix[2, 2] * db

        nr = (dnr ** (1 / 2.2)) * 255 if dnr >= 0 else -1
        ng = (dng ** (1 / 2.2)) * 255 if dng >= 0 else -1
        nb = (dnb ** (1 / 2.2)) * 255 if dnb >= 0 else -1

        return [self._clip255(nr), self._clip255(ng), self._clip255(nb)]

    def compute_rgb_ya(self, r: int, g: int, b: int):
        p = np.array([
            [0.99787176, 5.36027646, -0.21742974],
            [1.00219012, -1.60265749, 0.27315637],
            [0.98516759, 0.09213337, -2.06550866]
        ], dtype=float)

        dr = (r / 255.0) ** 2.2
        dg = (g / 255.0) ** 2.2
        db = (b / 255.0) ** 2.2

        ws = self.COVMatrix[0, 0] * dr + self.COVMatrix[0, 1] * dg + self.COVMatrix[0, 2] * db
        rg = self.COVMatrix[1, 0] * dr + self.COVMatrix[1, 1] * dg + self.COVMatrix[1, 2] * db
        yb = self.COVMatrix[2, 0] * dr + self.COVMatrix[2, 1] * dg + self.COVMatrix[2, 2] * db

        nr = (p[0, 0] * ws + p[0, 1] * rg + p[0, 2] * yb) / 0.2832
        ng = (p[1, 0] * ws + p[1, 1] * rg + p[1, 2] * yb) / 0.2777
        nb = (p[2, 0] * ws + p[2, 1] * rg + p[2, 2] * yb) / 0.2666

        sr = (nr ** (1 / 2.2)) * 255 if nr >= 0 else -1
        sg = (ng ** (1 / 2.2)) * 255 if ng >= 0 else -1
        sb = (nb ** (1 / 2.2)) * 255 if nb >= 0 else -1

        return [self._clip255(sr), self._clip255(sg), self._clip255(sb)]

    def create_image(self, new_path: str):
        img = Image.open(self.path).convert("RGBA")
        arr = np.array(img, dtype=np.uint8)

        h, w, _ = arr.shape
        out = arr.copy()

        for j in range(h):
            for i in range(w):
                r, g, b, a = arr[j, i]

                if self.model in (1, 2, 3):
                    cl = self.compute_rgb(int(r), int(g), int(b))
                elif self.model == 4:
                    cl = self.compute_rgb_ya(int(r), int(g), int(b))
                else:
                    cl = self.compute_rgb_new(int(r), int(g), int(b))

                out[j, i] = [cl[0], cl[1], cl[2], a]

        Image.fromarray(out, mode="RGBA").save(new_path)