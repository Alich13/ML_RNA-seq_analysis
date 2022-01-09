FROM continuumio/miniconda3

WORKDIR /app

# Create the environment:
COPY ./ .
COPY MY_env.yml .
RUN conda env create -f MY_env.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "ML_env", "/bin/bash", "-c"]

# Demonstrate the environment is activated:
RUN echo "Make sure flask is installed:"
RUN python -c "import keras"

# The code to run when container is started:
COPY test.py .
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "ML_env", "python", "run.py"]