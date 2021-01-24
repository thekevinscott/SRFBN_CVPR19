FROM python:3.8.6-slim-buster

RUN apt-get update -y

RUN apt-get install -y \
python3 python3-dev gcc \
curl vim wget git \
aptitude \
libgl1-mesa-glx \
libglib2.0-0 \
build-essential

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3 get-pip.py

RUN python3 -m pip install jupyterlab torch torchvision tqdm pillow scipy scikit-image opencv-python pandas onnx

EXPOSE 8888

WORKDIR /app

CMD python3 -m jupyter lab --ip 0.0.0.0 --allow-root
