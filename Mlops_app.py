from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
import torchvision.transforms as transforms
import io
import nest_asyncio
from Mlops_Model import LogisticRegression
import uvicorn


app = FastAPI()

# FIXME: Load your trained model here
model = LogisticRegression(input_dim=28*28, output_dim=10, hidden_layers=[128, 64], dropout_rate=0.2)
model.load_state_dict(torch.load("mnist_model.pth", map_location=torch.device("cpu")))
model.eval()

# Define the transformation pipeline
def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Transforms the input image to match model's input requirements."""
    transform_pipeline = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    return transform_pipeline(image).unsqueeze(0)


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert('L')  # Convert to grayscale
    # FIXME: Implement preprocessing of the image
    image = Image.open(io.BytesIO(await file.read()))
    image_tensor = preprocess_image(image)

    # FIXME: Implement model inference
    with torch.no_grad():
        output = model(image_tensor)
        predicted_class = output.argmax(dim=1).item()

    # FIXME: Return the prediction in the appropriate format
    return {"prediction": predicted_class}

nest_asyncio.apply()
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)