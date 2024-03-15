# LLM Data Talk

[streamlit-main-2024-03-15-00-03-95.webm](https://github.com/dmenezesgabriel/llm-data-talk/assets/50274255/ff25ba04-42f4-4fdc-858d-ef074125139f)

## chinook database

(github)[https://github.com/lerocha/chinook-database/releases]

## Development

- **set python version**:

```sh
pyenv global 3.9.6
```

- **create virtual environment**:

```sh
pyenv exec python -m venv venv
```

- **install requirements**:

```sh
pip install -r requirements-dev.txt
```

- **run**:

```sh
PYTHONDONTWRITEBYTECODE=1 streamlit run main.py
```

## Architecture

![clean-architecture](docs/assets/clean_architecture.jpg)
