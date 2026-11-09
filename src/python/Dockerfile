# Start from an official, lightweight Python image
FROM python:3.12-slim

# All following commands run inside /app in the container
WORKDIR /app

# Install packages first, so Docker can cache this layer:
# if only the script changes, packages don't need reinstalling
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the script into the container
COPY titanic.py .

# The dataset is NOT copied into the image. It is mounted at runtime
# into /app/data (see README), so the data never ends up in the image
# or the repo.

# Run the script when the container starts
CMD ["python", "titanic.py"]