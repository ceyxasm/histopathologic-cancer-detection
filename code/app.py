import streamlit as st
import pickle
import numpy as np
from PIL import Image
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.optim as optim

class Dataset(Dataset):
    def __init__(self, data_df, transform=None):
        super().__init__()
        self.df = data_df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img = self.df[index]
        image = np.array(Image.open(img))
        if self.transform is not None:
            image = self.transform(image)
        return image

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.avg = nn.AvgPool2d(8)
        self.fc = nn.Linear(512 * 1 * 1, 2)
        
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avg(x)
        x = x.view(-1, 512 * 1 * 1)
        x = self.fc(x)
        return x

Transformations = transforms.Compose([transforms.ToPILImage(),
                                  transforms.Pad(64, padding_mode='reflect'),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])
path_to_model='model/model.ckpt'
def preprocess(data):
    data = [data]
    dataset_inp = Dataset(data_df = data,transform = Transformations)
    load_inp = DataLoader(dataset = dataset_inp,batch_size=1)
    #image = Transformations(np.array(Image.open(data)))
    return load_inp


st.title("Histopathological Cancer Detection")
st.markdown("A site which detects Histopathological Cancer from .tif images using a Convoluted Neural Network, implemented with PyTorch!")

wav = st.file_uploader("Upload your Image file (TIF)",type = ['tif'])
if wav is not None:
    st.image(Image.open(wav),width = 300)
    wav = preprocess(wav)
    model = CNN()
    model.load_state_dict(torch.load(path_to_model,map_location=torch.device('cpu')))
    model.to(torch.device('cpu'))
    model.eval()
    ans = 0
    with torch.no_grad():
        for img in wav:
            img = img.to(torch.device('cpu'))
            _,ans = torch.max(model(img).data,1)
    #ans = st.write(model(torch.tensor([wav])))
    st.write('The prediction was','Yes' if ans == 1 else 'No')


#functions

# Sidebar to create multiple pages
#app_mode = st.sidebar.selectbox('Select Page',['Home','Exploratory Data Analysis'])
#if(app_mode == 'Home'):
    
#option = st.selectbox(
#'Select the machine learning model',
#('CNN Model','Dummy'))
#if(option == 'CNN Model'):
#    model = CNN()
#    model.load_state_dict(torch.load(path_to_model,map_location=torch.device('cpu')))
#    model.to(torch.device('cpu'))
#    model.eval()
#    ans = 0
#    with torch.no_grad():
#        for img in wav:
#            img = img.to(torch.device('cpu'))
#            _,ans = torch.max(model(img).data,1)
#    #ans = st.write(model(torch.tensor([wav])))
#    st.write('The prediction was','Yes' if ans == 1 else 'No')

    

   # if(option == 'LightGBM'):
   #         pass

   # elif(option == 'Random Forest Classifier'):
   #     pass



    

   # option = st.selectbox(
   #  'Features to train your model',
   #  ('Time Domain', 'Frequency Domain', 'Spectral Shape Based','All Features','Only MFCC'))

   # st.write('You selected:', option)
    

        

#elif(app_mode == 'Feature Visualisation'):
#    st.title("Visualisation")
#    st.header('Time Domain features')
#    st.subheader("Zero Crossing Rate")

    


















    




            
            

             





