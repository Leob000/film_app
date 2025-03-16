# Todo
- Rapport
- Sur quel dataset? Idem que l'article/Réduction de types d'objets/Réduction de résolution
- Quelle application? Quelle interactivité?
    - Visualisation des tSNE?
    - Appli qui tourne juste sur CPU laptop? Ou faire aussi une partie plus complexe?
        - Plutôt faire au moins un gros modèle efficace, pré-entraîné avec les poids importés dans streamlit, et un petit modèle entraînable laptop CPU même si réusltats dégeulasses
    - Idées interaction :
        - Visualisation des tSNE, visualisation de ce que le MLP voit, et lien entre les 2?
        - Un modèle classique, un modèle avec zero-shot ?
        - Question "fait main" avec un drop down menu pour poser des questions sur l'image

- Ne pas oublier de mettre un graph de comparaison de performance sur jeux de donnée classiques?
- Docu avec sphinx? Voir quel format de docstrings, extension vscode docstrings auto?
- Problème du training GPU poor:
    - Checkpointing: Make sure your training loop saves progress frequently so you can resume after a session timeout.
    - Mixed Precision: Leverage PyTorch's AMP to take advantage of the T4's tensor cores.

# Requirements
- Python 3.12
- Other dependencies listed in `requirements.txt`

# References
- The code in this repo is heavily inspired by the repos [Film](https://github.com/ethanjperez/film) and [Clever-iep](https://github.com/facebookresearch/clevr-iep)
- [Distill: Feature wise transformations](https://distill.pub/2018/feature-wise-transformations/)
- [Arxiv: FiLM: Visual Reasoning with a General Conditioning Layer](https://arxiv.org/pdf/1709.07871)

# Get the data
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
python run_model.py --program_generator <FiLM Generator filepath> --execution_engine <FiLMed Network filepath>
```