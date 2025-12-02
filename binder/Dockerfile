FROM mambaorg/micromamba:latest

# Needed for micromamba activation in Dockerfile RUN commands
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# Workdir inside container
WORKDIR /app

# Copy only env first to leverage layer caching
COPY binder/environment.yml /tmp/environment.yml

# Create the conda env
RUN micromamba env create -f /tmp/environment.yml && \
    micromamba clean --all --yes

# Now copy the rest of the project
COPY . /app

# Default environment name; adjust if changes in environment.yml
ENV CONDA_DEFAULT_ENV=gen_simpy_apps
ENV PATH=/opt/conda/envs/gen_simpy_apps/bin:$PATH

EXPOSE 8501

# Use shell-form so the env is active via MAMBA_DOCKERFILE_ACTIVATE
CMD ["streamlit", "run", "Home.py", "--server.address=0.0.0.0", "--server.port=8501"]
