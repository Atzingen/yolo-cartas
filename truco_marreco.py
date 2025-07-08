# from roboflow import Roboflow
# rf = Roboflow(api_key="kPvqRvHij3QHPhHFV2mx")
# project = rf.workspace("augmented-startups").project("playing-cards-ow27d")
# version = project.version(4)
# dataset = version.download("yolov11")

from ultralytics import YOLO

# Load a pre-trained YOLOv11n model
model = YOLO("yolo11n.pt")

# Path to your custom dataset's YAML file (assuming it's in a folder called "cards")
# Note: Roboflow downloaded YOLOv11 format should have a data.yaml file
dataset_yaml = "cards/data.yaml"

# Train the model on your custom dataset
results = model.train(
    data=dataset_yaml,
    epochs=10,
    imgsz=640,
    batch=512,
    patience=3,  # Early stopping patience
    save=True     # Save best model
)

# Validate the model
model.val()

# Example: Run inference on a single image
# results = model("cards/test/images/card_image.jpg")