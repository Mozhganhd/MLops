
# MNIST FastAPI Deployment

## Project Overview
This project demonstrates an end-to-end machine learning pipeline using the MNIST dataset. The main steps include:
- Training a Logistic Regression model on the MNIST dataset with PyTorch.
- Logging metrics and tracking the training process using Weights & Biases (wandb).
- Deploying the trained model via FastAPI for real-time predictions.
- Containerizing the FastAPI application using Docker for easy deployment.

## Files in This Repository
- **Mlops_Model.py**: Script for training the Logistic Regression model and saving it as `mnist_model.pth`.
- **Mlops_app.py**: FastAPI application to serve the model for predictions.
- **Dockerfile**: Docker configuration for containerizing the FastAPI application.
- **requirements.txt**: List of dependencies required to run the project.
- **README.md**: Project overview and instructions for setup and usage.

## Prerequisites
- Python 3.8 or higher
- [Docker](https://www.docker.com/get-started) installed for containerization
- [Weights & Biases (wandb)](https://wandb.ai/) account (optional, only needed if you want to log training metrics)

## Instructions for Running the Code Locally

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/Mozhganhd/MLops.git
   cd MLops
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Train the model:
   - Run the training script to train the Logistic Regression model and save the model weights to `mnist_model.pth`:
     ```bash
     python Mlops_Model.py
     ```
   - This will also log metrics to wandb if you’re logged in.

4. Run the FastAPI application:
   - After training, you can start the FastAPI application for model inference:
     ```bash
     uvicorn Mlops_app:app --host 0.0.0.0 --port 8000
     ```
   - Access the application at `http://localhost:8000`.

### Using the FastAPI Application
- Once the app is running, go to `http://localhost:8000/docs` to view and interact with the API documentation.
- You can upload an image of a handwritten digit to the `/predict/` endpoint to get a prediction.

## Instructions for Building and Running the Docker Container

### Build the Docker Image
1. Ensure `mnist_model.pth` is in the project directory (created after running `Mlops_Model.py`).
2. Build the Docker image:
   ```bash
   docker build -t fastapi-mnist-app .
   ```

### Run the Docker Container
1. Start the container and expose port 8000:
   ```bash
   docker run -p 8000:8000 fastapi-mnist-app
   ```
2. Access the FastAPI application at `http://localhost:8000`.

## Weights & Biases (wandb) Report
For training metrics and model performance logs, check out the wandb report here: [W&B Project URL](https://wandb.ai/your-username/project-name)

## Project Structure

```
MLops/
│
├── Mlops_Model.py          # Script to train and save the model
├── Mlops_app.py            # FastAPI application for model inference
├── mnist_model.pth         # Saved model weights (generated after training)
├── Dockerfile              # Docker configuration for containerization
├── requirements.txt        # Dependencies required to run the project
└── README.md               # Project overview and setup instructions
```