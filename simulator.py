from cvd_matrix import CVDMatrix
from cvd_filter import CVDFilter


def simulate_image(image_path: str,
                   output_path: str,
                   model: int,
                   type_cdo: int,
                   degree: float,
                   matrix_file: str = "cvd_matrices.json",
                   excel_path: str = "covmartixlmsrgb.xlsx"):
    cvdm = CVDMatrix(
        model=model,
        type_cdo=type_cdo,
        degree=degree,
        matrix_file=matrix_file
    )
    matrix = cvdm.compute()

    cvdf = CVDFilter(
        matrix=matrix,
        image_path=image_path,
        model=model,
        excel_path=excel_path
    )
    cvdf.create_image(output_path)