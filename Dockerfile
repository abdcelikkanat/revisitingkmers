FROM nvcr.io/nvidia/pytorch:23.04-py3

WORKDIR /workdir

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .
