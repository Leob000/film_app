# Todo
- [-] Rapport
- Streamlit
    - [ ] Docu sphinx
    - [ ] Gros modèle pré-entraîné
        - [x] Obtention des weights
        - [ ] Streamlit poser questions sur image
        - [ ] Visualisation des histogrammes gamma/beta
        - [ ] Visualisation tSNE
        - [ ] Visualisation de ce que le MLP "voit"
    - [ ] Petit modèle, train sur CPU
        - Avoir aussi le preprocessing réduit?
        - Comment avoir un temps d'entraînement rapide? réduire architecture? réduire train/val dataset?
        - [ ] Streamlit train
        - [ ] Streamlit questions
- Bonus:
    - Zero-shot
    - Graph comparaison de performance sur jeux de donnée classique

# Requirements
- Python 3.12
- Other dependencies listed in `requirements.txt`

# References
- The code in this repo is heavily inspired by the repos [Film](https://github.com/ethanjperez/film) and [Clever-iep](https://github.com/facebookresearch/clevr-iep)
- [Distill: Feature wise transformations](https://distill.pub/2018/feature-wise-transformations/)
- [Arxiv: FiLM: Visual Reasoning with a General Conditioning Layer](https://arxiv.org/pdf/1709.07871)

# Get the data
For each script, check the `.sh` and/or the `.py` associated file to modify parameters.
To download the data, run:
```bash
mkdir data
wget https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip -O data/CLEVR_v1.0.zip
unzip data/CLEVR_v1.0.zip -d data
```

To preprocess the data from pngs to a h5 file for each train/val/test set, run the following code. The data will be the raw pixels, there are options to extract features with the option `--model resnet101` (1024x14x14 output), or to set a maximum number of X processed images `--max_images X` (check `extract_features.py`).
```bash
sh scripts/extract_features.sh
```

To preprocess the questions, execute this script:
```bash
sh scripts/preprocess_questions.sh
```

To train the model:
```bash
sh scripts/train/film.sh
```

To run the model (on `CLEVR_val_000017.png` by default):
```bash
sh scripts/run_model.sh
```

# Get the small streamlit model data
To create images, questions and answer in order to train the small model on streamlit, run
```bash
sh faklevr_scripts/small_faklevr_dataset_creation.sh
```

To preprocess images :
```bash
sh faklevr_scripts/faklevr_extract_features_raw.sh
```

To preprocess questions :
```bash
sh faklevr_scripts/faklevr_preprocess_questions.sh
```