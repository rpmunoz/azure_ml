import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core import Workspace

from azureml.opendatasets import MNIST
from utils import load_data

configFolder = '../config'
dataFolder = '../data/mnist'
resultsFolder = '../results'

print("Setting up service principal")
sp = ServicePrincipalAuthentication(tenant_id=os.environ['AML_TENANT_ID'],
                                    service_principal_id=os.environ['AML_PRINCIPAL_ID'],
                                    service_principal_password=os.environ['AML_PRINCIPAL_PASS'])

print('Setting up workspace')
ws = Workspace.from_config(path=configFolder, auth=sp)

print('Downloading MNIST data')
os.makedirs(dataFolder, exist_ok=True)
mnistFileDataset = MNIST.get_file_dataset()
mnistFileDataset.download(dataFolder, overwrite=True)

mnistFileDataset = mnistFileDataset.register(workspace=ws,
                                             name='mnist_dataset',
                                             description='training and test dataset',
                                             create_new_version=True)

# note we also shrink the intensity values (X) from 0-255 to 0-1. This helps the model converge faster.
X_train = load_data(glob.glob(os.path.join(dataFolder, "**/train-images-idx3-ubyte.gz"), recursive=True)[0], False) / 255.0
X_test = load_data(glob.glob(os.path.join(dataFolder, "**/t10k-images-idx3-ubyte.gz"), recursive=True)[0], False) / 255.0
y_train = load_data(glob.glob(os.path.join(dataFolder, "**/train-labels-idx1-ubyte.gz"), recursive=True)[0], True).reshape(-1)
y_test = load_data(glob.glob(os.path.join(dataFolder, "**/t10k-labels-idx1-ubyte.gz"), recursive=True)[0], True).reshape(-1)

# now let's show some randomly chosen images from the traininng set.
count = 0
sample_size = 30
plt.figure(figsize=(16, 6))
for i in np.random.permutation(X_train.shape[0])[:sample_size]:
    count = count + 1
    plt.subplot(1, sample_size, count)
    plt.axhline('')
    plt.axvline('')
    plt.text(x=10, y=-10, s=y_train[i], fontsize=18)
    plt.imshow(X_train[i].reshape(28, 28), cmap=plt.cm.Greys)

os.makedirs(resultsFolder, exist_ok=True)
figFile = os.path.join(resultsFolder, 'mnist_sample.jpg')
plt.savefig(figFile)

