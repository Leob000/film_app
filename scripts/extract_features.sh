python scripts/extract_features.py \
  --input_image_dir data/CLEVR_v1.0/images/train \
  --output_h5_file data/train_features_raw.h5

python scripts/extract_features.py \
  --input_image_dir data/CLEVR_v1.0/images/val \
  --output_h5_file data/val_features_raw.h5

python scripts/extract_features.py \
  --input_image_dir data/CLEVR_v1.0/images/test \
  --output_h5_file data/test_features_raw.h5