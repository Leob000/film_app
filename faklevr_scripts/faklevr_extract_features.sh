python faklevr_scripts/faklevr_extract_features.py \
  --input_image_dir "data/faklevr/images/train" \
  --output_h5_file data/faklevr/train_features.h5 \
  --model resnet101

python faklevr_scripts/faklevr_extract_features.py \
  --input_image_dir "data/faklevr/images/val" \
  --output_h5_file data/faklevr/val_features.h5 \
  --model resnet101

python faklevr_scripts/faklevr_extract_features.py \
  --input_image_dir "data/faklevr/images/test" \
  --output_h5_file data/faklevr/test_features.h5 \
  --model resnet101