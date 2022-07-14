FROM nvidia/cuda:11.7.0-runtime-ubuntu18.04

# install utilities
RUN apt-get update && \
    apt-get install --no-install-recommends -y curl

ENV CONDA_AUTO_UPDATE_CONDA=false \
    PATH=/opt/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py38_4.9.2-Linux-x86_64.sh \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -b -p /opt/miniconda \
    && rm ~/miniconda.sh \
    && sed -i "$ a PATH=/opt/miniconda/bin:\$PATH" /etc/environment

# Installing python dependencies
RUN python3 -m pip --no-cache-dir install --upgrade pip && \
    python3 --version && \
    pip3 --version

# Get the Fast API image with Python version 3.7
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7    

# Create the directory for the container
WORKDIR /app
COPY requirements.txt ./requirements.txt

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app.py ./

# Copy the serialized model
COPY ./models/VGG16_v4_class_weights_export.pt ./models/VGG16_v4_class_weights_export.pt

# Run by specifying the host and port
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]