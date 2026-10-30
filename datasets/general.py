from datasets import mnist


def get_loaders(dataset_name, batch_size=128, transform=None):
    if dataset_name == "mnist":
        return mnist.get_mnist_loader(batch_size, transform)
    
    return None

def get_datasets(dataset_name, transform=None):
    if dataset_name == "mnist":
        return mnist.get_mnist_dataset(transform)
    
    return None

