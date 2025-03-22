python faklevr_scripts/faklevr_preprocess_questions.py \
  --input_questions_json data/faklevr/questions/faklevr_train_question.json \
  --output_h5_file data/faklevr/train_questions.h5 \
  --output_vocab_json data/faklevr/vocab.json

python faklevr_scripts/faklevr_preprocess_questions.py \
  --input_questions_json data/faklevr/questions/faklevr_val_question.json \
  --output_h5_file data/faklevr/val_questions.h5 \
  --input_vocab_json data/faklevr/vocab.json

python faklevr_scripts/faklevr_preprocess_questions.py \
  --input_questions_json data/faklevr/questions/faklevr_test_question.json \
  --output_h5_file data/faklevr/test_questions.h5 \
  --input_vocab_json data/faklevr/vocab.json