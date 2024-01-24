# En Poisson Backend

## Installation Instructions

Download poetry. You can either download it from the website (https://python-poetry.org/docs/#installation) or use pip:

```bash
pip install poetry
```

Then, inside of the repo folder, run:

```bash
poetry install
```

which should install all of the dependencies that you need.

## Running

To run the server, run:

```bash
poetry run uvicorn main:app --reload
```

or

```bash
poetry shell
uvicorn main:app --reload
```

Eventually we could upgrade to docker which might make some of this easier.
