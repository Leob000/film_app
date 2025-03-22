python faklevr_scripts/faklevr_extract_features.py \
  --input_image_dir "data/faklevr/images/train" \
  --output_h5_file data/faklevr/train_features_raw.h5 \
  --image_height 56 \
  --image_width 56 \
  --model none

python faklevr_scripts/faklevr_extract_features.py \
  --input_image_dir "data/faklevr/images/val" \
  --output_h5_file data/faklevr/val_features_raw.h5 \
  --image_height 56 \
  --image_width 56 \
  --model none

python faklevr_scripts/faklevr_extract_features.py \
  --input_image_dir "data/faklevr/images/test" \
  --output_h5_file data/faklevr/test_features_raw.h5 \
  --image_height 56 \
  --image_width 56 \
  --model none