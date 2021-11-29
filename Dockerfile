FROM nvcr.io/nvidia/tensorflow:18.08-py3
WORKDIR /my-ml-files
RUN pip install jupyter
EXPOSE 8888
RUN pip install keras
