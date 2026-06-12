from torch import get_device
import torch
import src.saliency
import src.trainer
import src.CNN
import src.utils
import config
import src.autoencoder
from src.dataset import get_dataloaders
from torchvision import datasets, transforms
import src.saliency


#src.trainer.run_training();

model = src.CNN.MnistCNN();
src.utils.load_model( model ) # i get some saved model
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
dataset = datasets.MNIST(config.DATA_DIR, train=False,  download=True, transform=transform) # I load the dataset
number = 3000 # random number
image, label = dataset[number]
#src.trainer.save_model_visual_result( model , image.unsqueeze(0) , "img " + str(number) ) #unsqueeze bc (1,1,28,28) is required

#plausability = src.autoencoder.ClassConditionalPlausibility()
#train , _ = get_dataloaders()
#plausability.load_saved_autoencoders()
#plausability.train_all( train , src.utils.get_device() );
#plausability.save_trained_autoencoders()


image = image.unsqueeze(0) # one_image_saliency requires tensor(1,1,28,28) and image is (1,28,28)
for i in range(5):
    saliency_map = src.saliency.one_image_saliency( model , image , 8 ) # gerenate saliency map
    src.saliency.save_saliency_result( image , saliency_map , "saliency_" + str(number) + "_" + str(i+1) ) # save the map
    image = src.utils.merge_image_saliency( image , saliency_map ) # make new image by adding saliency
    if( image.dim() == 3 ):
        image = image.unsqueeze(0)
print( model(image) )