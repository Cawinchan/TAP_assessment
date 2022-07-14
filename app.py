# Load the libraries
from fastapi import FastAPI,UploadFile
import torch
from PIL import Image
from torchvision import transforms
from io import BytesIO

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_ft = torch.jit.load('models/VGG16_v4_class_weights_export.pt')
model_ft.to(device)

# Initialize an instance of FastAPI
app = FastAPI()

# Allow for the support of uploading files
def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image

@app.get("/ping")
def root():
    return {"message": "Welcome to Our Ship Type Classification FastAPI!"}

@app.post("/infer")
async def predict_image(file: UploadFile):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
        
    image = read_imagefile(await file.read())

    # Follow the same pipeline as training phase
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
        transforms.RandomAffine(degrees=40, translate=None, scale=(1, 2), shear=15, resample=False, fillcolor=0),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    image = transform(image)
    image = image.unsqueeze(0) # add one dimension to the front to account for batch_size

    with torch.no_grad():
        outputs = model_ft(image.to(device))
        _, preds = torch.max(outputs, 1)
        print(int(preds))

    return {
            "boat_class_prediction": int(preds), 
            }