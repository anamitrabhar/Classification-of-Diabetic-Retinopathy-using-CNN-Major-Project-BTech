"""Create an Image Classification Web App using PyTorch and Streamlit."""
# import libraries
from PIL import Image
from torchvision import models, transforms
import torch
import streamlit as st
import numpy as np


# set title of app
st.title("Diabetic Retinopathy Classification Application")
st.write("")

# enable users to upload images for the model to make predictions
file_up = st.file_uploader("Upload an image", type = "png")


def predict(image):
    """Return top 5 predictions ranked by highest probability.
    Parameters
    ----------
    :param image: uploaded image
    :type image: jpg
    :rtype: list
    :return: top 5 predictions ranked by highest probability
    """
    # create a ResNet model
    resnet = torch.load('finalDRclassifier.pth',map_location=torch.device('cpu'))
    #resnet = models.resnet101(pretrained = True)

    # transform the input image through resizing, normalization
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
            )])

    # load the image, pre-process it, and make predictions
    img = Image.open(image)
    batch_t = torch.unsqueeze(transform(img), 0)
    resnet.eval()
    out = resnet(batch_t)
    print(out)

    classes=['0','1','2','3','4']
    # return the top 5 predictions ranked by highest probabilities
    prob = torch.nn.functional.softmax(out, dim = 1)[0] * 100
    _, indices = torch.sort(out, descending = True)
    return [(classes[idx], prob[idx].item()) for idx in indices[0][:2]]




def image_loader(loader, image):

    image = loader(image).float()
    image = torch.tensor(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image

data_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])


def use_model(image):
    model_ft = torch.load('finalDRclassifier.pth', map_location=torch.device('cpu'))
    model_ft.eval()
    out = model_ft(image_loader(data_transforms, image)).detach().numpy()
    out = torch.from_numpy(out)
    classes=['0','1','2','3','4']
    # return the top 5 predictions ranked by highest probabilities
    prob = torch.nn.functional.softmax(out, dim = 1)[0] * 100
    _, indices = torch.sort(out, descending = True)
    return [(classes[idx], prob[idx].item()) for idx in indices[0][:1]]

if file_up is not None:
    image = Image.open(file_up)
    st.image(image, caption = 'Uploaded Image.', use_column_width = True)
    st.write("")
    st.write("Clasifying.. Just a second ...")
    labels = use_model(image)

    for i in labels:
        st.write("Prediction ->  DR Level", i[0], ",   Score: ", round(i[1],3))


    #print( np.argmax(model_ft(image_loader(data_transforms, image)).detach().numpy()))
    #print(model_ft(image_loader(data_transforms, image)).detach().numpy())
