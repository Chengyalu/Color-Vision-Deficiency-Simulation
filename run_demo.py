from simulator import simulate_image


if __name__ == "__main__":
    simulate_image(
        image_path="test.png",
        output_path="newPicture.png",
        model=1,
        type_cdo=0,
        degree=0.8,
        excel_path="covmartixlmsrgb.xlsx"
    )