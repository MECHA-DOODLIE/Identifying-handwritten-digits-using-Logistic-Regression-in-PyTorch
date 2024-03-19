# Identifying-handwritten-digits-using-Logistic-Regression-in-PyTorch
This project implements logistic regression in PyTorch for classifying handwritten digits from the MNIST dataset. Logistic regression is a fundamental machine learning algorithm used for binary classification tasks. In this project, we extend its application to multi-class classification by leveraging PyTorch.

### Key Features:

- Utilizes the MNIST dataset, a benchmark dataset in the field of computer vision.

- Implements logistic regression model architecture using PyTorch's neural network module.

- Demonstrates data preprocessing techniques such as normalization and data augmentation.

- Performs model training and evaluation, achieving an accuracy of approximately 82% on the test set.

- Includes hyperparameter tuning for optimizing model performance.

  ## Installation
Firstly, you will need to install PyTorch into your Python environment. The easiest way to do this is to use the pip or conda tool. Visit pytorch.org and install the version of your Python interpreter and the package manager that you would like to use.

1. Install PyTorch by following the instructions at [pytorch.org](https://pytorch.org).
2. Clone this repository:

```bash
git clone https://github.com/MECHA-DOODLIE/handwritten-digit-recognition.git
```
With PyTorch installed, let us now have a look at the code. Write the three lines given below to import the required library functions and objects.
```
import torch 
import torch.nn as nn 
import torchvision.datasets as dsets 
import torchvision.transforms as transforms 
from torch.autograd import Variable
```
Here, the torch.nn module contains the code required for the model, torchvision.datasets contain the MNIST dataset. It contains the dataset of handwritten digits that we shall be using here. The torchvision.transforms module contains various methods to transform objects into others. Here, we shall be using it to transform from images to PyTorch tensors. Also, the torch.autograd module contains the Variable class amongst others, which will be used by us while defining our tensors.

Next, we shall download and load the dataset to memory.
```
# MNIST Dataset (Images and Labels) 
train_dataset = dsets.MNIST(root ='./data',  
                            train = True,  
                            transform = transforms.ToTensor(), 
                            download = True) 
  
test_dataset = dsets.MNIST(root ='./data',  
                           train = False,  
                           transform = transforms.ToTensor()) 
  
# Dataset Loader (Input Pipeline) 
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,  
                                           batch_size = batch_size,  
                                           shuffle = True) 
  
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,  
                                          batch_size = batch_size,  
                                          shuffle = False)
```
Now, we shall define our hyperparameters.
```
# Hyper Parameters  
input_size = 784
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001
```
In our dataset, the image size is 28*28. Thus, our input size is 784. Also, 10 digits are present in this and hence, we can have 10 different outputs. Thus, we set num_classes as 10. Also, we shall train five times on the entire dataset. Finally, we will train in small batches of 100 images each so as to prevent the crashing of the program due to memory overflow.
After this, we shall be defining our model as below. Here, we shall initialize our model as a subclass of torch.nn.Module and then define the forward pass. In the code that we are writing, the softmax is internally calculated during each forward pass and hence we do not need to specify it inside the forward() function.
```
class LogisticRegression(nn.Module): 
    def __init__(self, input_size, num_classes): 
        super(LogisticRegression, self).__init__() 
        self.linear = nn.Linear(input_size, num_classes) 
  
    def forward(self, x): 
        out = self.linear(x) 
        return out 
```
Having defined our class, now we instantiate an object for the same.
```
model = LogisticRegression(input_size, num_classes)
```
Next, we set our loss function and the optimizer. Here, we shall be using the cross-entropy loss and for the optimizer, we shall be using the stochastic gradient descent algorithm with a learning rate of 0.001 as defined in the hyperparameter above.
```
criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
```
Now, we shall start the training. Here, we shall be performing the following tasks:

1. Reset all gradients to 0.
2. Make a forward pass.
3. Calculate the loss.
4. Perform backpropagation.
5. Update all weights.
```
# Training the Model 
for epoch in range(num_epochs): 
    for i, (images, labels) in enumerate(train_loader): 
        images = Variable(images.view(-1, 28 * 28)) 
        labels = Variable(labels) 
  
        # Forward + Backward + Optimize 
        optimizer.zero_grad() 
        outputs = model(images) 
        loss = criterion(outputs, labels) 
        loss.backward() 
        optimizer.step() 
  
        if (i + 1) % 100 == 0: 
            print('Epoch: [% d/% d], Step: [% d/% d], Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, 
                     len(train_dataset) // batch_size, loss.data[0]))
```
Finally, we shall be testing out the model by using the following code.
```
# Test the Model 
correct = 0
total = 0
for images, labels in test_loader: 
    images = Variable(images.view(-1, 28 * 28)) 
    outputs = model(images) 
    _, predicted = torch.max(outputs.data, 1) 
    total += labels.size(0) 
    correct += (predicted == labels).sum() 
  
print('Accuracy of the model on the 10000 test images: % d %%' % ( 
            100 * correct / total))
```
Assuming that you performed all steps correctly, you will get an accuracy of 82%, which is far off from today’s state-of-the-art model, which uses a special type of neural network architecture.


