StrengthApp
==============================

An AI-powered mobile application that can serve as your strength training coach, guiding you through exercises like squats, bench presses, or deadlifts. Initialiy, based on ongoing research, our application aims to provide a straightforward method for predicting 1-RM using the built-in camera of your smartphone. Stay tuned!

-----------
[One Repetiton Maximum Research Paper](https://github.com/MateuszKunik/StrengthCoach/blob/master/1RM_ResearchPaper.pdf)

The article presents an innovative approach to predicting the One-Repetition Maximum (in short 1-RM) parameter, commonly used in strength sports. The proposed method combines deep neural networks with computer vision techniques. The study focuses on a comprehensive analysis of strength exercise technique based on audiovisual materials. Traditional methods of measuring 1-RM and their limitations are discussed, alongside a newly proposed method that integrates advanced image analysis with athletes' training specifics.

In the study, computer vision algorithms were applied to estimate body pose, followed by the use of advanced neural network architectures to analyze the obtained data. A dedicated dataset containing recordings of study participants was developed. The goal of this method is to increase the accuracy of 1-RM prediction while eliminating the inconveniences of traditional methods. The article discusses potential practical applications and future research directions in applying artificial intelligence to strength sports.



Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
