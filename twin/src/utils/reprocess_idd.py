from prepare_data import convert_split

for split in ["train", "val"]:
    convert_split(
        img_dir=f"data/raw/idd20k_lite/leftImg8bit/{split}",
        mask_dir=f"data/raw/idd20k_lite/gtFine/{split}",
        lane_dir=None,
        output_dir="data/processed",
        split_name=split,
        dataset_type="idd_lite",
    )