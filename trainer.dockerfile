# Base image
FROM python:3.11-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# copy relevant files of the project
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY mlops_soloproject/ mlops_soloproject/
COPY data/ data/
COPY models/ models/

# install dependencies
WORKDIR /
RUN --mount=type=cache,target=~/pip/.cache pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

# define entrypoint
ENTRYPOINT ["python", "-u", "models/train_model.py"]