import torch.nn as nn
import torch.nn.functional as F

# define the CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.checkpoint_file = 'model_scratch_v2.pt'
        self.epoch = 1
        self.min_validation_loss = float('inf')
        self.max_accuracy = 0.
        
        input_dims = 32
        conv_out_dims = 4 # 3 maxpool of 2
        conv_out_depth = 64
        self.num_labels = 133        
        
        self.linear_input_dims = conv_out_depth * conv_out_dims**2
        print("Conv input wxh {} \nLinear input len {}".format(input_dims, self.linear_input_dims))
        
        # Feature extractor
                            # in, out, kernel
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, conv_out_depth, 3, padding=1)

        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(self.linear_input_dims, 500)
        self.fc2 = nn.Linear(500, self.num_labels)

        # dropout layer
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # flatten image input
        x = x.view(-1, self.linear_input_dims)
        # add dropout layer
        x = self.dropout(x)
        # add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # add 2nd hidden layer, with relu activation function
        x = self.fc2(x)
        return x