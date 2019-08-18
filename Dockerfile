FROM tensorflow/tensorflow:1.12.0-gpu-py3
RUN apt-get update -qq && apt-get install -yqq --no-install-recommends vim && apt-get install -yqq gcc
RUN pip install --no-cache-dir nni==0.7.1
RUN pip install --no-cache-dir Pillow==2.2.2
RUN pip install --no-cache-dir scikit-learn==0.20.1
WORKDIR "/"
