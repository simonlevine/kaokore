
import joblib
from loguru import logger
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import torchvision.models as models
from tqdm.notebook import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.externals import joblib
import torch
from PIL import Image

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():

    model_fname = 'fitted_knn_kaokore.sav'    
    # Fit kNN model on embedded images
    k = 5
    image_embedder = Embedder()
    print('Loading images...')
    image_embedder.load_images()
    print('Embedding Kaokore paintings with a neural network...')
    image_embedder.embed_images()
    print(f'Fitting and saving {k}-nearest-neighbour model on image embeddings...')
    image_embedder.generate_estimator()

    inferrer = Inference(model_fname)
    test_embedding=inferrer.embed_single_image()


class Inference:
    def __init__(self,model_fname):
        self.embedder=Embedder()
        self.imsize=256
        self.loader = transforms.Compose([transforms.Scale(self.imsize), transforms.ToTensor()])
        self.knn_estimator = joblib.load(model_fname)

    def embed_single_image(self,image_name):
        """load image, returns cuda tensor"""
        image = Image.open(image_name)
        image = self.loader(image).float()
        image = Variable(image, requires_grad=True)
        image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
        image = image.to(DEVICE)
        embedding = self.embedder.model(image).detach().numpy().flatten()
        return embedding

    def regress_single_embedding(self):
        self.knn_estimator.tr



class Embedder:
    def __init__(self):
        self.image_embeddings=[]
        self.model = models.resnet18(pretrained=True).to(DEVICE)

    def load_images(self):
        self.loader = DataLoader(ImageFolder(root='../kaokore',transform=torchvision.transforms.ToTensor()))

    def embed_images(self):
        for x,_ in self.loader:
            x = x.to(DEVICE)
            fx = self.model(x)
            self.embeddings_flattened.append(fx.detach().cpu().numpy().flatten())

    def generate_estimator(self):
        knn_estimator = NearestNeighbors(n_neighbors=5, metric="cosine")
        knn_estimator.fit(np.array(self.image_embeddings))
        # save the model to disk
        joblib.dump(knn_estimator, open(model_fname, 'wb'))


if __name__=='__main__':
    main()
