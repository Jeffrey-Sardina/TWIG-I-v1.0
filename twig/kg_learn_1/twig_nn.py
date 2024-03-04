'''
==========================
Neural Network Definitions
==========================
'''
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torcheval.metrics.functional import r2_score
import torch.nn.functional as F

'''
===============
Reproducibility
===============
'''
torch.manual_seed(17)

'''
==========================
Neural Network Definitions
==========================
'''
class TWIG_KGL_v0(nn.Module):
    def __init__(self, n_global, n_local):
        '''
        init() init creates the neural network object and initialises its
        parameters. It also defines all layers used in learned. For an overview
        of the neural architecture, please see comments on the forward()
        function.

        The arguments it accepts are:
            - n_global (int) the number of global graph structural features
              that are present in the input features vectors.. These will be on
              the left (lower-index) side of the input features vectors.
            - n_local (int)the number of local graph structural features
              that are present in the input features vectors.. These will be on
              the right (higher-index) side of the input features vectors.

        The values it returns are:
            - None (init function to create an object)
        '''
        super().__init__()
        self.n_global = n_global #14
        self.n_local = n_local #22

        # struct parts are from the version with no hps included
        # we now want to cinclude hps, hwoever
        self.linear_struct_1 = nn.Linear(
            in_features=n_local,
            out_features=10
        )
        self.relu_1 = nn.ReLU()
        self.dropout_1 = nn.Dropout(p=1e-2)
        
        self.linear_struct_2 = nn.Linear(
            in_features=10,
            out_features=10
        )
        self.relu_2 = nn.ReLU()
        self.dropout_2 = nn.Dropout(p=1e-2)

        self.linear_final = nn.Linear(
            in_features=10,
            out_features=1
        )
        self.sigmoid_final = nn.Sigmoid()

    def forward(self, X):
        '''
        forward() defines the forward pass of the NN and the neural
        architecture. For TWIG KGL v0, this achitecture is approximately as
        follows.
            - all global features are dropped and ignored.
            - all local features are passed through three dense layers
              separated by ReLU activation.

        **NOTE** Since this version drops global structural features, it is
        possible that its performance will suffer if used to train on many
        different KGs at once, where global structure may be relevant. However,
        this has not yet been empirically determiend.

        The arguments it accepts are:
            - X (Tensor): a tensor with feature vectors as rows, and as many
              rows as the batch size that is in use.

        The values it returns are:
            - X (Tensor): tensor with the same number of rows as the input, but
              only one value in each row, wich represents the score of the
              triple that was described by that row's feature vector.
        '''
        _, X_local = X[:, :self.n_global], X[:, self.n_global:]

        X_local = self.linear_struct_1(X_local)
        X_local = self.relu_1(X_local)
        X_local = self.dropout_1(X_local)

        X_local = self.linear_struct_2(X_local)
        X_local = self.relu_2(X_local)
        X_local = self.dropout_2(X_local)

        X = self.linear_final(X_local)
        X = self.sigmoid_final(X) #maybe use softmax instead,but I like this for now theoretically

        return X
    
class TWIG_KGL_v1(nn.Module):
    def __init__(self, n_global, n_local):
        '''
        init() init creates the neural network object and initialises its
        parameters. It also defines all layers used in learned. For an overview
        of the neural architecture, please see comments on the forward()
        function.

        The arguments it accepts are:
            - n_global (int) the number of global graph structural features
              that are present in the input features vectors.. These will be on
              the left (lower-index) side of the input features vectors.
            - n_local (int)the number of local graph structural features
              that are present in the input features vectors.. These will be on
              the right (higher-index) side of the input features vectors.

        The values it returns are:
            - None (init function to create an object)
        '''
        super().__init__()
        self.n_global = n_global #14
        self.n_local = n_local #22

        # struct parts are from the version with no hps included
        # we now want to cinclude hps, hwoever
        self.linear_struct_1 = nn.Linear(
            in_features=n_local,
            out_features=10
        )
        self.relu_1 = nn.ReLU()
        
        self.linear_struct_2 = nn.Linear(
            in_features=10,
            out_features=10
        )
        self.relu_2 = nn.ReLU()

        self.linear_global_1 = nn.Linear(
            in_features=n_global,
            out_features=6
        )
        self.relu_3 = nn.ReLU()

        self.linear_integrate_1 = nn.Linear(
            in_features=6 + 10,
            out_features=8
        )
        self.relu_4 = nn.ReLU()

        self.linear_final = nn.Linear(
            in_features=8,
            out_features=1
        )
        self.sigmoid_final = nn.Sigmoid()

    def forward(self, X):
        '''
        forward() defines the forward pass of the NN and the neural
        architecture. For TWIG KGL v1, this achitecture is approximately as
        follows.
            - global and local features are split apart.
            - local features are rrun through 2 dense layers.
            - global features are run through one dense layer.
            - all features are concatenated and run through two dense layers.
            - note that all layers are separated by a ReLU activation function.

        **NOTE** Since this version uses global structural features, it is
        expected that it will perform better when training TWIG on multiple KGs
        at the same time. However, this has not yet been empirically
        determinned.

        The arguments it accepts are:
            - X (Tensor): a tensor with feature vectors as rows, and as many
              rows as the batch size that is in use.

        The values it returns are:
            - X (Tensor): tensor with the same number of rows as the input, but
              only one value in each row, wich represents the score of the
              triple that was described by that row's feature vector.
        '''
        X_global, X_local = X[:, :self.n_global], X[:, self.n_global:]

        X_local = self.linear_struct_1(X_local)
        X_local = self.relu_1(X_local)

        X_local = self.linear_struct_2(X_local)
        X_local = self.relu_2(X_local)

        X_global = self.linear_global_1(X_global)
        X_global = self.relu_3(X_global)

        X = self.linear_integrate_1(
            torch.concat(
                [X_local, X_global],
                dim=1
            ),
        )
        X = self.relu_4(X)

        X = self.linear_final(X)
        X = self.sigmoid_final(X)

        return X
