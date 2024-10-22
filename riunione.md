# Appunti riunione 03/10/2024

## Nuovo approccio
Nuova idea di Esposito e Mirko....
Contrastive loss. 
Addestrare una Feed Forward con pochi livelli densi con RELU. Passare al modello 3 esempi per
volta di cui dato un esempio:
- il secondo è simile
- il terzo è completamente diverso. 
---

### Contrastive Loss
Semplificare notevolmente il modello usando una FF come oggetto di base. La dimensione dell'output può essere quella che vogliamo. Più è alta, meglio è.
Modello:
- Input: 768
- Output: 300

Si tratta di mappare le triple in input in uno spazio vettoriale diverso rispetto quello
di partenza, usando un margine da massimizzare.
L'approccio suggerisce un metodo simile alle SVM. Qui usato per fare clustering. 


## Un esempio:
Let's go through a simple example using a feedforward neural network and the contrastive loss.
Suppose we want to cluster the MNIST dataset of handwritten digits into 10 clusters, one for each digit class. We'll use a simple feedforward neural network with two hidden layers to learn a representation of each digit image.
Here's a simple PyTorch implementation:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

# Define the neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)  # input layer (28x28 images) -> hidden layer (128 units)
        self.fc2 = nn.Linear(128, 128)  # hidden layer (128 units) -> hidden layer (128 units)
        self.fc3 = nn.Linear(128, 10)  # hidden layer (128 units) -> output layer (10 units)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # activation function for hidden layer
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

# Define the contrastive loss function
def contrastive_loss(anchor, positive, negative, margin=2.0):
    distance_positive = torch.pairwise_distance(anchor, positive)
    distance_negative = torch.pairwise_distance(anchor, negative)
    return torch.clamp(distance_positive - distance_negative + margin, min=0).mean()

# Train the network using contrastive loss
criterion = contrastive_loss
optimizer = optim.Adam(net.parameters(), lr=0.001)

for epoch in range(10):  # loop over the dataset multiple times
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = inputs.view(-1, 784)  # flatten the input data

        # Create triplets for contrastive loss
        anchor = inputs[0::3]  # anchor points
        positive = inputs[1::3]  # positive points (same class as anchor)
        negative = inputs[2::3]  # negative points (different class from anchor)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs_anchor = net(anchor)
        outputs_positive = net(positive)
        outputs_negative = net(negative)

        # Compute contrastive loss
        loss = criterion(outputs_anchor, outputs_positive, outputs_negative)

        # Backward pass
        loss.backward()

        # Update the model parameters
        optimizer.step()

        # Print loss at each 100 mini-batches
        if i % 100 == 0:
            print('Epoch {}: Loss = {:.4f}'.format(epoch+1, loss.item()))
```
In this example, we define a neural network with two hidden layers and an output layer with 10 units (one for each digit class). We then define the contrastive loss function, which takes three inputs: the anchor point, the positive point, and the negative point. During training, we create triplets of points by selecting anchor points, positive points (same class as anchor), and negative points (different class from anchor). We then pass these points through the network and compute the contrastive loss.

### Using the learned network for clustering
Once the network is trained, we can use it to cluster new, unseen data points. Here's an example:


```python
# Load a new, unseen image
new_image = ...

# Preprocess the image
new_image = transforms.ToTensor()(new_image)

# Pass the image through the network
outputs = net(new_image)

# Get the cluster assignment
cluster_assignment = torch.argmax(outputs)

print('Cluster assignment:', cluster_assignment.item())
```

In this example, we load a new, unseen image and pass it through the network. The output of the network is a vector of 10 values, one for each digit class. We then use the `argmax` function to get the index of the maximum value, which corresponds to the cluster assignment.
Note that the cluster assignment is not necessarily the same as the digit class label. The cluster assignment is a learned representation of the data, which may not necessarily correspond to the original class labels. I hope this example helps clarify how to use the contrastive loss for clustering! Let me know if you have any further questions.


---
### Tipi di loss
- Triplet loss
- Contrastive loss

Sui gruppi di nomi di un certo IBAN:
- per ogni gruppo:
  - prendere 2 esempi di quel gruppo
  - prendere un esempio fuori da quel gruppo
  - applicare la loss contrastiva
  - ripetere
 