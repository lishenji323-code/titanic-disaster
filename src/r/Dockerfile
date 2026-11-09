# Start from an official R image with a fixed R version
FROM rocker/r-ver:4.4.1

# All following commands run inside /app in the container
WORKDIR /app

# Install packages first, so Docker can cache this layer:
# if only the script changes, packages don't need reinstalling
COPY install_packages.R .
RUN Rscript install_packages.R

# Copy the analysis script into the container
COPY titanic.R .

# The dataset is NOT copied into the image. It is mounted at runtime
# into /app/data (see README), so the data never ends up in the image
# or the repo.

# Run the script when the container starts
CMD ["Rscript", "titanic.R"]