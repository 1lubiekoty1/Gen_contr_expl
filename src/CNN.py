# src/model.py

from typing import Sequence
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import Sequential, Softmax
from torch.nn.modules.activation import Sigmoid
import config

class ConvolutionLayer( nn.Module ):
    def __init__( self , in_channels , out_channels , kernel_size , pooling_size , padding ):
        super().__init__()
        layers = []
        layers.append( nn.Conv2d( in_channels , out_channels , kernel_size = kernel_size , padding = padding ))
        layers.append( nn.ReLU() )
        layers.append( nn.MaxPool2d( pooling_size , pooling_size ))
        self.block = nn.Sequential(*layers)
    def forward( self , x ):
        return self.block(x)

class LinearLayer( nn.Module ):
    def __init__( self , in_features , out_features , dropout_rate ):
        super().__init__()
        layers = []
        layers.append( nn.Linear(in_features , out_features) )
        layers.append( nn.ReLU() )
        layers.append( nn.Dropout(dropout_rate))
        self.block = nn.Sequential(*layers)
    def forward( self , x ):
        return self.block(x)


class MnistCNN(nn.Module):
    def __init__(self):
        super(MnistCNN, self).__init__()
        #CONVOLUTION
        layers = []
        for i in range(len(config.CONV_LAYERS) -1 ):
            layers.append( ConvolutionLayer( config.CONV_LAYERS[i] , config.CONV_LAYERS[i+1] , 3 , 2 , 1 ) )

        #CONNECT CONVOLUTION TO LINEAR
        layers.append( nn.Flatten( 1 , -1 ) )

        #HIDDEN LINEAR
        for i in range(len(config.LINEAR_LAYERS) -1 ):
            layers.append( LinearLayer( config.LINEAR_LAYERS[i] , config.LINEAR_LAYERS[i+1] , config.DROPOUT_RATE ) )

        #HEAD
        layers.append( nn.Linear( config.LINEAR_LAYERS[-1] , config.NUM_CLASSES ) )
        layers.append( nn.Softmax() )

        self.block = nn.Sequential( *layers )

    def forward( self , x ):
        return self.block( x ) 