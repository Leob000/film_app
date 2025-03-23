# Todo
- [-] Rapport
- Streamlit
    - [ ] Docu sphinx
    - [ ] 2 Pages différentes, attendre voir si dataset custom pour changer
    - [x] wget simple pour les poids
    - [ ] Gros modèle pré-entraîné
        - [x] Obtention des weights
        - [x] Streamlit poser questions sur image
        - [x] Visualisation de ce que le MLP "voit"
        - [x] Phrases exemples, mots que le modèle connaît
        - [ ] Visualisation des histogrammes gamma/beta
        - [ ] Visualisation tSNE
    - [ ] Petit modèle, train sur CPU
        - [ ] Streamlit train
        - [ ] Streamlit questions
    - [ ] Dataset custom?

# Simple use
## Requirements
### Python and packages
- Python 3.12
- Other dependencies listed in `requirements.txt`, run `pip install -r requirements.txt`.

### Download our pre-trained model for the CLEVR dataset
We pretrained a big model (3 FiLM layers, resnet101 with 1024 feature maps for the vision CNN model).
To get the weights (in `data/best.pt`), run:

```bash
wget "https://www.dropbox.com/scl/fi/1exvuj8mp0122c0faogte/best.pt?rlkey=huyzf4nhnr6p8jwsnyiy14nd0&st=odj3a2ns" -O data/best.pt
```

### To train a new model on our custom simpler dataset
To create images, questions and answers in order to train the small model on streamlit, run:
```bash
sh faklevr_scripts/favlevr_bundle.sh
```

Train the small streamlit model (it takes around 20-25 minutes on CPU):
```bash
sh scripts/train/film_faklevr_raw.sh
```

## Streamlit app
To launch the streamlit app, run:
```bash
streamlit run Hello.py
```

# Detailed use
## CLEVR Dataset
If you wish to run the models in the terminal and modify parameters, follow these instructions.

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

# References
- The code in this repo is heavily inspired by the repos [Film](https://github.com/ethanjperez/film) and [Clever-iep](https://github.com/facebookresearch/clevr-iep)
- [Distill: Feature wise transformations](https://distill.pub/2018/feature-wise-transformations/)
- [Arxiv: FiLM: Visual Reasoning with a General Conditioning Layer](https://arxiv.org/pdf/1709.07871)