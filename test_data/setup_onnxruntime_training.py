'''This script generates the assets required to test onnxruntime training'''

from pathlib import Path
from IPython.core.display import Image, display
import onnxruntime.training.artifacts as artifacts
import numpy as np
import os
import glob
import json
import shutil
import torchvision
import torch
import onnx
import requests
from zipfile import ZipFile
import tarfile
import argparse

onnxruntime_version = "1.18.0"
dir = os.path.abspath(os.path.dirname(__file__))
platform = "linux-x64"

# Preprocess the images and convert to tensors as expected by the model
# Makes the image a square and resizes it to 224x224 as is expected by
# the mobilenetv2 model
# Normalize the image by subtracting the mean (0.485, 0.456, 0.406) and
# dividing by the standard deviation (0.229, 0.224, 0.225)
def image_file_to_tensor(file):
    from PIL import Image
    image = Image.open(file)
    width, height = image.size
    if width > height:
        left = (width - height) // 2
        right = (width + height) // 2
        top = 0
        bottom = height
    else:
        left = 0
        right = width
        top = (height - width) // 2
        bottom = (height + width) // 2
    image = image.crop((left, top, right, bottom)).resize((224, 224))
    pix = np.transpose(np.array(image, dtype=np.float32), (2, 0, 1))
    pix = pix / 255.0
    pix[0] = (pix[0] - 0.485) / 0.229
    pix[1] = (pix[1] - 0.456) / 0.224
    pix[2] = (pix[2] - 0.406) / 0.225
    return pix

def get_processed_image(file):
    image_tensor = image_file_to_tensor(file)
    return {"array": image_tensor.tolist()}

def download_images():
    url="https://github.com/microsoft/onnxruntime-training-examples/raw/master/on_device_training/mobile/android/java/data/images.zip"
    r = requests.get(url, stream=True)
    with open(Path(dir, "onnxruntime_training_test", "images.zip"), 'wb') as fd:
        for chunk in r.iter_content(chunk_size=128):
            fd.write(chunk)
            
def download_onnx_training_lib():
    url = f"https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-training-{platform}-{onnxruntime_version}.tgz"
    r = requests.get(url, stream=True)
    with open(Path(dir, f"onnxruntime-training-{platform}.tgz"), 'wb') as fd:
        for chunk in r.iter_content(chunk_size=128):
            fd.write(chunk)
    tar = tarfile.open(Path(dir, f"onnxruntime-training-{platform}.tgz"), "r:gz")
    tar.extractall(dir)
    tar.close()
    os.rename(Path(dir, f"onnxruntime-training-{platform}-{onnxruntime_version}/lib/libonnxruntime.so.{onnxruntime_version}"),
              Path(dir, f"onnxruntime_training_test/onnxruntime_training.so"))
    shutil.rmtree(Path(dir, f"onnxruntime-training-{platform}-{onnxruntime_version}"))
    os.remove(Path(dir, f"onnxruntime-training-{platform}.tgz"))

def generate_training_artifacts():
    base = Path(dir, "onnxruntime_training_test", "training_artifacts")
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(exist_ok=True, parents=True)
    model = torchvision.models.mobilenet_v2(
    weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V2)

    # The original model is trained on imagenet which has 1000 classes.
    # For our image classification scenario, we need to classify among 4 categories.
    # So we need to change the last layer of the model to have 4 outputs.
    model.classifier[1] = torch.nn.Linear(1280, 4)

    # Export the model to ONNX.
    model_name = "mobilenetv2"
    torch.onnx.export(model, torch.randn(1, 3, 224, 224),
                    Path(dir, "onnxruntime_training_test", "training_artifacts", f"{model_name}.onnx").__str__(),
                    input_names=["input"], output_names=["output"],
                    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})

    # Load the onnx model.
    onnx_model = onnx.load(Path(dir, "onnxruntime_training_test", "training_artifacts", f"{model_name}.onnx"))

    # Define the parameters that require their gradients to be computed
    # (trainable parameters) and those that do not (frozen/non trainable parameters).
    requires_grad = ["classifier.1.weight", "classifier.1.bias"]
    frozen_params = [
    param.name
    for param in onnx_model.graph.initializer
    if param.name not in requires_grad
    ]

    # Generate the training artifacts.
    artifacts.generate_artifacts(
    onnx_model,
    requires_grad=requires_grad,
    frozen_params=frozen_params,
    loss=artifacts.LossType.CrossEntropyLoss,
    optimizer=artifacts.OptimType.AdamW,
    artifact_directory=Path(dir, "onnxruntime_training_test", "training_artifacts"))

def load_dataset_files():
    animals = {
        "dog": [],
        "cat": [],
        "elephant": [],
        "cow": []
    }
    for animal in animals:
        animals[animal] = glob.glob(
            Path(dir, "onnxruntime_training_test", "images", f"{animal}", "*").__str__())
        animals[animal] = sorted(animals[animal])
    return animals

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnxruntime", required=False)
    parser.add_argument("--platform", required=False)
    args = parser.parse_args()
    
    if args.onnxruntime:
        onnxruntime_version = args.onnxruntime
    if args.platform:
        platform = args.platform
    
    print(f"generating testing assets for onnxruntime training {onnxruntime_version} and platform {platform}")
    test_assets_path = Path(dir, "onnxruntime_training_test")
    if test_assets_path.exists():
        shutil.rmtree(test_assets_path)
    test_assets_path.mkdir(parents=True)
    print("0. Getting onnxruntime training lib")
    download_onnx_training_lib()
    print("1. downloading training images...")
    download_images()
    with ZipFile(Path(test_assets_path, "images.zip")) as zObject:
        # into a specific location. 
        zObject.extractall(path=test_assets_path)  
    print("2. generating processed images...")
    dataset = load_dataset_files()
    for animal in dataset:
        animal_folder = Path(test_assets_path, "processed_images", animal)
        animal_folder.mkdir(parents=True, exist_ok=True)
        for image_file in dataset[animal]:
            image_processed = get_processed_image(image_file)
            filename = Path(animal_folder, os.path.basename(os.path.splitext(image_file)[0]) + ".json")
            with open(filename, "w") as f:
                json.dump(image_processed, f)
    print("3. generating training assets...")
    generate_training_artifacts()
    print("Setup done")