import torch
import torchvision
from matplotlib import pyplot as plt

def predict_custom_image(image_path,model,transform,class_names):
    custom_image=torchvision.io.read_image(image_path).type(torch.float32)
    transformed_custom_image=transform(custom_image)
    model.eval()
    with torch.inference_mode():
        prediction=model(transformed_custom_image.unsqueeze(0)).argmax(dim=1)
    plt.imshow(custom_image.permute(1,2,0)/255)
    plt.title(class_names[prediction])