import torch.nn as nn
import torch.nn.functional as F

# define the CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.checkpoint_file = 'model_scratch_v3.pt'
        self.epoch = 1
        self.min_validation_loss = float('inf')
        self.max_accuracy = 0.
        
        input_dims = 64
        conv_out_dims = 4 # 64->32->16->8->4
        conv_out_depth = 64
        self.num_labels = 133        
        
        self.linear_input_dims = conv_out_depth * conv_out_dims**2
        print("Conv input wxh {} \nLinear input len {}".format(input_dims, self.linear_input_dims))
        
        # Feature extractor
                            # in, out, kernel
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, conv_out_depth, 3, padding=1)
        self.conv4 = nn.Conv2d(conv_out_depth, conv_out_depth, 3, padding=1)


        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
#         hidden_nodes = int((self.linear_input_dims+self.num_labels)/2)
        self.fc1 = nn.Linear(self.linear_input_dims, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, self.num_labels)       

        # dropout layer
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = self.pool(F.leaky_relu(self.conv3(x)))
        x = self.pool(F.leaky_relu(self.conv4(x)))        
        # flatten image input
        x = x.view(-1, self.linear_input_dims)
        
        # Classifier
        x = self.dropout(x)
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.dropout(x)

        x = self.fc3(x)
        return x