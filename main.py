from torch import get_device
import src.trainer
import src.CNN
import src.utils
import config
import src.autoencoder
from src.dataset import get_dataloaders
from torchvision import datasets, transforms


src.trainer.run_training();

#model = src.CNN.MnistCNN();
#src.utils.load_model( model ) # i get some saved model
#transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
#dataset = datasets.MNIST(config.DATA_DIR, train=False,  download=True, transform=transform) # I load the dataset
#image, label = dataset[97] # a random image
#src.trainer.save_model_visual_result( model , image.unsqueeze(0) , "img" ) #unsqueeze for some reason

plausability = src.autoencoder.ClassConditionalPlausibility()
train , _ = get_dataloaders()
plausability.load_saved_autoencoders()
plausability.train_all( train , src.utils.get_device() );
plausability.save_trained_autoencoders()
