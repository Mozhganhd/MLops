# Use an official lightweight Python image as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the necessary files
COPY Mlops_Model.py /app/Mlops_Model.py
COPY Mlops_app.py /app/Mlops_app.py
COPY requirements.txt /app/requirements.txt
COPY mnist_model.pth /app/mnist_model.pth

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that FastAPI will run on
EXPOSE 8000

# Set the command to run the FastAPI application
CMD ["uvicorn", "Mlops_app:app", "--host", "0.0.0.0", "--port", "8000"]