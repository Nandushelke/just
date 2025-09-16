Datasets

Datasets are NOT bundled in this repository due to size and licensing. Use this guide to obtain and place them under the `data/` directory.

General placement

- Create a subfolder per dataset, e.g., `data/imdb/`, `data/sentiment140/`, `data/fer2013/`, `data/utkface/`, `data/ravdess/`.
- Keep original splits/file names where possible. You can add your own processed versions under `data/processed/<dataset_name>/`.

Text: Sentiment

- IMDB Movie Reviews: [Keras datasets](https://keras.io/api/datasets/imdb/), [Hugging Face](https://huggingface.co/datasets/stanfordnlp/imdb)
  - Place under: `data/imdb/`
- Sentiment140 (Twitter): [Official](https://www.sentiment140.com/), [Hugging Face](https://huggingface.co/datasets/sentiment140)
  - Place under: `data/sentiment140/`

Emotion: Text + Speech

- ISEAR (text): Availability varies; a derivative is on [Kaggle](https://www.kaggle.com/datasets/ishantjuyal/emotions-in-text)
  - Place under: `data/isear/`
- RAVDESS (speech): [Zenodo DOI](https://doi.org/10.5281/zenodo.1188976)
  - Place under: `data/ravdess/`

Facial Emotion Recognition (Images)

- FER-2013: [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
  - Place CSV/images under: `data/fer2013/`

Age Prediction (Faces)

- UTKFace: [Project page](https://susanqq.github.io/UTKFace/)
  - Place images under: `data/utkface/`
- Adience: [Project page](https://talhassner.github.io/home/projects/Adience/Adience-data.html)
  - Place images/lists under: `data/adience/`

Tips

- Many of these are available via the Hugging Face `datasets` library for quick prototyping (e.g., IMDB, Sentiment140). If you prefer that route, install `datasets` and load within notebooks/scripts; save processed artifacts into `data/processed/`.
- Kaggle downloads require a Kaggle account and API credentials. See Kaggle CLI docs to configure `~/.kaggle/kaggle.json` and use `kaggle datasets download ...`.
- Always review and comply with dataset licenses and terms of use.

