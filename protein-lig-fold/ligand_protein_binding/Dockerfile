FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN pip install torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cu117.html

COPY . .

CMD ["python", "run.py"]
