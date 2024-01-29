# Use a base image with Python 3.9.
FROM python:3.9


# Set working directory in the container.
WORKDIR /opt/binary_cnn_classifier

RUN mkdir src
RUN mkdir data

# copy application code
COPY ./src ./src
COPY ./data ./data
COPY main.py .

# Install dependencies.
RUN pip install --no-cache-dir \
    keras \
    matplotlib \
    numpy \
    wget \
    scipy \
    tensorflow

# Run the application.
CMD ["python", "main.py"]