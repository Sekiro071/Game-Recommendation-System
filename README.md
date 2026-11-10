# Respawned: Game Recommendation System

Respawned is a content-based game recommendation app that uses TF-IDF and cosine similarity to suggest games with similar metadata.

## Project structure

```text
.
├── data/                 # Local dataset used by the app
├── notebooks/            # Exploratory analysis notebook
├── src/
│   └── game_recommendation/
│       ├── __init__.py
│       └── app.py        # Gradio application and recommendation engine
├── pyproject.toml        # Project metadata and dependencies
└── README.md
```

## Setup

Requires Python 3.11 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

## Run the app

From the repository root:

```powershell
python -m game_recommendation.app
```

Or, after installing the package:

```powershell
game-recommendation
```

The app loads `data/games.csv` relative to the project and does not depend on a machine-specific file path.

## Dataset

The included dataset contains game names, genres, categories, SteamSpy tags, developers, and image URLs. If the dataset is replaced, keep the expected column names used by the recommendation engine.

## Notebook

The exploratory notebook is available at `notebooks/game_recommendation_exploration.ipynb`.
