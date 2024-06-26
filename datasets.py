import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import Dataset
from torchvision import datasets
import random
import transforms

import pickle
import os

known_class = -1
init_percent = -1


class CustomCIFAR100Dataset_train(Dataset):

    
    def __init__(self, root="./data/cifar100", train=True, download=True, transform=None, invalidList=None):

        self.cifar100_dataset = datasets.CIFAR100(root, train=train, download=download, transform=transform)
        self.targets = self.cifar100_dataset.targets

        if invalidList is not None:

            targets = np.array(self.cifar100_dataset.targets)
            targets[targets >= known_class] = known_class
            self.cifar100_dataset.targets = targets.tolist()

    def __getitem__(self, index):
        data_point, label = self.cifar100_dataset[index]
        return index, (data_point, label)

    def __len__(self):
        return len(self.cifar100_dataset)

class CustomCIFAR100Dataset_test(Dataset):
    cifar100_dataset = None
    targets = None
    @classmethod
    def load_dataset(cls, root="./data/cifar100", train=False, download=True, transform=None):
        cls.cifar100_dataset = datasets.CIFAR100(root, train=train, download=download, transform=transform)
        cls.targets = cls.cifar100_dataset.targets

    def __init__(self):
        if CustomCIFAR100Dataset_test.cifar100_dataset is None:
            raise RuntimeError("Dataset not loaded. Call load_dataset() before creating instances of this class.")

    def __getitem__(self, index):
        data_point, label = CustomCIFAR100Dataset_test.cifar100_dataset[index]
        return index, (data_point, label)

    def __len__(self):
        return len(CustomCIFAR100Dataset_test.cifar100_dataset)


class CustomMNISTDataset_train(Dataset):

    def __init__(self, root="./data/mnist", train=True, download=True, transform=None, invalidList=None):

        self.mnist_dataset = datasets.MNIST(root, train=train, download=download, transform=transform)
        self.targets = self.mnist_dataset.targets

        if invalidList is not None:
            targets = np.array(self.mnist_dataset.targets)
            targets[targets >= known_class] = known_class
            self.mnist_dataset.targets = targets.tolist()

    def __getitem__(self, index):
        data_point, label = self.mnist_dataset[index]
        return index, (data_point, label)

    def __len__(self):
        return len(self.mnist_dataset)


class CustomMNISTDataset_test(Dataset):
    mnist_dataset = None
    targets = None

    @classmethod
    def load_dataset(cls, root="./data/mnist", train=False, download=True, transform=None):
        cls.mnist_dataset = datasets.MNIST(root, train=train, download=download, transform=transform)
        cls.targets = cls.mnist_dataset.targets

    def __init__(self):
        if CustomMNISTDataset_test.mnist_dataset is None:
            raise RuntimeError("Dataset not loaded. Call load_dataset() before creating instances of this class.")

    def __getitem__(self, index):
        data_point, label = CustomMNISTDataset_test.mnist_dataset[index]
        return index, (data_point, label)

    def __len__(self):
        return len(CustomMNISTDataset_test.mnist_dataset)

"""This code defines a class constructor that sets up data loaders for a machine learning task. The data loaders 
handle loading of data from a dataset, applying transformations to the data, and batching the data for training or 
testing a model.

Here's a general overview of what the code does:

1. **Data Transformation**: It first defines a series of transformations to be applied to the images in the dataset. 
These transformations include converting the images to grayscale, converting them to PyTorch tensors, and normalizing 
them.

2. **Memory Management**: It sets a flag for whether to pin memory or not. Pinning memory can speed up data transfer 
between CPU and GPU.

3. **Data Indexing**: It checks if there's a list of invalid indices. If there is, it adds these indices to the list 
of labeled indices. This could be used to exclude certain data points from the training process.

4. **Dataset and DataLoader Setup**: It creates a custom MNIST dataset with the defined transformations and invalid 
list. Depending on the flags `is_filter` and `is_mini`, it sets up different ways to split the data into labeled and 
unlabeled sets. It then creates PyTorch DataLoaders for the training and testing datasets. The DataLoaders handle 
batching of the data and shuffling.

5. **Data Filtering**: It defines methods to filter the data based on the class labels. This could be used in a 
scenario where you have known and unknown classes, and you want to separate the data accordingly.

The code is designed to be flexible to different scenarios, such as whether to use a GPU, whether to filter the data, 
and whether to use a smaller version of the dataset."""
class MNIST(object):
    def __init__(self, batch_size, use_gpu, num_workers, is_filter, is_mini, unlabeled_ind_train=None,
                 labeled_ind_train=None, invalidList=None):
        transform = transforms.Compose([
            # Convert the images to grayscale with 3 output channels
            transforms.Grayscale(num_output_channels=3),
            # Convert the images to PyTorch tensors
            transforms.ToTensor(),
            # Normalize the images with mean and standard deviation
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        pin_memory = True if use_gpu else False

        if invalidList is not None:
            labeled_ind_train = labeled_ind_train + invalidList

        trainset = CustomMNISTDataset_train(transform=transform, invalidList=invalidList)

        if unlabeled_ind_train == None and labeled_ind_train == None:
            if is_mini:
                labeled_ind_train, unlabeled_ind_train = self.filter_known_unknown_10percent(trainset)
                self.labeled_ind_train, self.unlabeled_ind_train = labeled_ind_train, unlabeled_ind_train
            else:
                labeled_ind_train = self.filter_known_unknown(trainset)
                self.labeled_ind_train = labeled_ind_train
        else:
            self.labeled_ind_train, self.unlabeled_ind_train = labeled_ind_train, unlabeled_ind_train

        if is_filter:
            print("openset here!")
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory,
                sampler=SubsetRandomSampler(labeled_ind_train),
            )
            unlabeledloader = torch.utils.data.DataLoader(
                trainset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory,
                sampler=SubsetRandomSampler(unlabeled_ind_train),
            )
        else:
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=pin_memory,
            )

        CustomCIFAR100Dataset_test.load_dataset(transform=transform)
        testset = CustomCIFAR100Dataset_test()
        filter_ind_test = self.filter_known_unknown(testset)
        self.filter_ind_test = filter_ind_test

        if is_filter:
            testloader = torch.utils.data.DataLoader(
                testset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory,
                sampler=SubsetRandomSampler(filter_ind_test),
            )
        else:
            testloader = torch.utils.data.DataLoader(
                testset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory,
            )

        self.trainloader = trainloader
        if is_filter: self.unlabeledloader = unlabeledloader
        self.testloader = testloader
        self.num_classes = known_class

    def filter_known_unknown(self, dataset):
        filter_ind = []
        for i in range(len(dataset.targets)):
            c = dataset.targets[i]
            if c < known_class:
                filter_ind.append(i)
        return filter_ind

    def filter_known_unknown_10percent(self, dataset):
        filter_ind = []
        unlabeled_ind = []
        for i in range(len(dataset.targets)):
            c = dataset.targets[i]
            if c < known_class:
                filter_ind.append(i)
            else:
                unlabeled_ind.append(i)

        # 随机选
        random.shuffle(filter_ind)
        labeled_ind = filter_ind[:len(filter_ind) * init_percent // 1000]
        unlabeled_ind = unlabeled_ind + filter_ind[len(filter_ind) * init_percent // 1000:]
        return labeled_ind, unlabeled_ind


class CIFAR100(object):
    def __init__(self, batch_size, use_gpu, num_workers, is_filter, is_mini, unlabeled_ind_train=None,
                 labeled_ind_train=None, invalidList=None):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])

        pin_memory = True if use_gpu else False

        if invalidList is not None:
            labeled_ind_train = labeled_ind_train + invalidList

        #CustomCIFAR100Dataset_train.load_dataset(transform=transform_train)
        # trainset = torchvision.datasets.CIFAR100("./data/cifar100", train=True, download=True,
        #                                          transform=transform_train)

        trainset = CustomCIFAR100Dataset_train(transform=transform_train, invalidList=invalidList)

        ## 初始化
        if unlabeled_ind_train == None and labeled_ind_train == None:
            if is_mini:
                labeled_ind_train, unlabeled_ind_train = self.filter_known_unknown_10percent(trainset)
                self.labeled_ind_train, self.unlabeled_ind_train = labeled_ind_train, unlabeled_ind_train
            else:
                labeled_ind_train = self.filter_known_unknown(trainset)
                self.labeled_ind_train = labeled_ind_train
        else:

            self.labeled_ind_train, self.unlabeled_ind_train = labeled_ind_train, unlabeled_ind_train

        if is_filter:
            print("openset here!")
            

            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory,
                sampler=SubsetRandomSampler(labeled_ind_train),
            )


            unlabeledloader = torch.utils.data.DataLoader(
                trainset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory,
                sampler=SubsetRandomSampler(unlabeled_ind_train),
            )
        else:
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=pin_memory,
            )

        # testset = torchvision.datasets.CIFAR100("./data/cifar100", train=False, download=True, transform=transform_test)
        CustomCIFAR100Dataset_test.load_dataset(transform=transform_test)
        testset = CustomCIFAR100Dataset_test()

        filter_ind_test = self.filter_known_unknown(testset)
        self.filter_ind_test = filter_ind_test

        if is_filter:
            testloader = torch.utils.data.DataLoader(
                testset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory,
                sampler=SubsetRandomSampler(filter_ind_test),
            )
        else:
            testloader = torch.utils.data.DataLoader(
                testset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory,
            )

        self.trainloader = trainloader
        if is_filter: self.unlabeledloader = unlabeledloader
        self.testloader = testloader
        self.num_classes = known_class

    def filter_known_unknown(self, dataset):
        filter_ind = []
        for i in range(len(dataset.targets)):
            c = dataset.targets[i]
            if c < known_class:
                filter_ind.append(i)
        return filter_ind

    def filter_known_unknown_10percent(self, dataset):
        filter_ind = []
        unlabeled_ind = []

        for i in range(len(dataset.targets)):
            c = dataset.targets[i]
            if c < known_class:
                filter_ind.append(i)
            else:
                unlabeled_ind.append(i)

        print ("Shuffle")
        # 随机选
        random.shuffle(filter_ind)
        labeled_ind = filter_ind[:len(filter_ind) * init_percent // 100]
        unlabeled_ind = unlabeled_ind + filter_ind[len(filter_ind) * init_percent // 100:]
        return labeled_ind, unlabeled_ind




class CustomCIFAR10Dataset_train(Dataset):
    # cifar100_dataset = None
    # targets = None

    # @classmethod
    # def load_dataset(cls, root="./data/cifar100", train=True, download=True, transform=None):
    #    cls.cifar100_dataset = datasets.CIFAR100(root, train=train, download=download, transform=transform)
    #    cls.targets = cls.cifar100_dataset.targets

    def __init__(self, root="./data/cifar10", train=True, download=True, transform=None, invalidList=None):
        # if CustomCIFAR100Dataset_train.cifar100_dataset is None:
        #    raise RuntimeError("Dataset not loaded. Call load_dataset() before creating instances of this class.")

        self.cifar10_dataset = datasets.CIFAR10(root, train=train, download=download, transform=transform)
        self.targets = self.cifar10_dataset.targets

        if invalidList is not None:
            targets = np.array(self.cifar10_dataset.targets)
            targets[targets >= known_class] = known_class
            self.cifar10_dataset.targets = targets.tolist()

    def __getitem__(self, index):
        data_point, label = self.cifar10_dataset[index]
        return index, (data_point, label)

    def __len__(self):
        return len(self.cifar10_dataset)


class CustomCIFAR10Dataset_test(Dataset):
    cifar10_dataset = None
    targets = None

    @classmethod
    def load_dataset(cls, root="./data/cifar10", train=False, download=True, transform=None):
        cls.cifar10_dataset = datasets.CIFAR10(root, train=train, download=download, transform=transform)
        cls.targets = cls.cifar10_dataset.targets

    def __init__(self):
        if CustomCIFAR10Dataset_test.cifar10_dataset is None:
            raise RuntimeError("Dataset not loaded. Call load_dataset() before creating instances of this class.")

    def __getitem__(self, index):
        data_point, label = CustomCIFAR10Dataset_test.cifar10_dataset[index]
        return index, (data_point, label)

    def __len__(self):
        return len(CustomCIFAR10Dataset_test.cifar10_dataset)






class CIFAR10(object):

    def __init__(self, batch_size, use_gpu, num_workers, is_filter, is_mini, unlabeled_ind_train=None,
                 labeled_ind_train=None, invalidList=None):

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        pin_memory = True if use_gpu else False

        if invalidList is not None:
            labeled_ind_train = labeled_ind_train + invalidList

        #trainset = torchvision.datasets.CIFAR10("./data/cifar10", train=True, download=True, transform=transform_train)
        trainset = CustomCIFAR10Dataset_train(transform=transform_train, invalidList=invalidList)


        ## 初始化
        if unlabeled_ind_train == None and labeled_ind_train == None:
            if is_mini:
                labeled_ind_train, unlabeled_ind_train = self.filter_known_unknown_10percent(trainset)
                self.labeled_ind_train, self.unlabeled_ind_train = labeled_ind_train, unlabeled_ind_train
            else:
                labeled_ind_train = self.filter_known_unknown(trainset)
                self.labeled_ind_train = labeled_ind_train
        else:
            self.labeled_ind_train, self.unlabeled_ind_train = labeled_ind_train, unlabeled_ind_train

        if is_filter:
            print("openset here!")
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory,
                sampler=SubsetRandomSampler(labeled_ind_train),
            )
            unlabeledloader = torch.utils.data.DataLoader(
                trainset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory,
                sampler=SubsetRandomSampler(unlabeled_ind_train),
            )
        else:
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=pin_memory,
            )

        #testset = torchvision.datasets.CIFAR10("./data/cifar10", train=False, download=True, transform=transform_test)
        
        CustomCIFAR10Dataset_test.load_dataset(transform=transform_test)
        testset = CustomCIFAR10Dataset_test()

        filter_ind_test = self.filter_known_unknown(testset)
        self.filter_ind_test = filter_ind_test

        if is_filter:
            testloader = torch.utils.data.DataLoader(
                testset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory,
                sampler=SubsetRandomSampler(filter_ind_test),
            )
        else:
            testloader = torch.utils.data.DataLoader(
                testset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory,
            )

        self.trainloader = trainloader
        if is_filter: self.unlabeledloader = unlabeledloader
        self.testloader = testloader
        self.num_classes = known_class

    def filter_known_unknown(self, dataset):
        filter_ind = []
        for i in range(len(dataset.targets)):
            c = dataset.targets[i]
            if c < known_class:
                filter_ind.append(i)
        return filter_ind

    def filter_known_unknown_10percent(self, dataset):
        filter_ind = []
        unlabeled_ind = []
        for i in range(len(dataset.targets)):
            c = dataset.targets[i]
            if c < known_class:
                filter_ind.append(i)
            else:
                unlabeled_ind.append(i)

        # 随机选
        random.shuffle(filter_ind)
        labeled_ind = filter_ind[:len(filter_ind) * init_percent // 100]
        unlabeled_ind = unlabeled_ind + filter_ind[len(filter_ind) * init_percent // 100:]
        return labeled_ind, unlabeled_ind



class CustomTinyImageNetDataset_train(Dataset):
    def __init__(self, root='./data/tiny-imagenet-200', train=True, download=True,target_train=None, transform=None, invalidList=None):
        self.tiny_imagenet_dataset = datasets.ImageFolder(os.path.join(root, 'train' if train else 'val'),
                                                          transform=transform)
        self.targets = target_train


        if invalidList is not None:
            targets = np.array(self.targets)
            targets[targets >= known_class] = known_class
            self.targets = targets.tolist()

    def __getitem__(self, index):
        data_point, _ = self.tiny_imagenet_dataset[index]
        # data_point, _ = self.tiny_imagenet_dataset[index]
        # label = self.targets[index]
        label = self.targets[index]
        return index, (data_point, label)

    def __len__(self):
        return len(self.tiny_imagenet_dataset)



class CustomTinyImageNetDataset_test(Dataset):
    tiny_imagenet_dataset = None
    targets = None
    image_to_label = None
    label_to_index = None

    @classmethod
    def load_dataset(cls, root='./data/tiny-imagenet-200', split='val', transform=None):
        cls.image_to_label = parse_val_annotations(root)
        cls.tiny_imagenet_dataset = datasets.ImageFolder(f'{root}/{split}', transform=transform)
        cls.label_to_index = {label: index for index, label in enumerate(sorted(set(cls.image_to_label.values())))}

        # Custom sorting function to sort images by their numerical part
        def sort_key(item):
            file_name = os.path.basename(item[0])
            file_number = int(file_name.split('.')[0].split('_')[1])
            return file_number

        # Sort the images by their file names using the custom sorting function
        cls.tiny_imagenet_dataset.imgs.sort(key=sort_key)

        cls.targets = []
        for img_file, _ in cls.tiny_imagenet_dataset.imgs:
            img_base_name = os.path.basename(img_file)
            img_label = cls.image_to_label[img_base_name]
            img_index = cls.label_to_index[img_label]
            cls.targets.append(img_index)

    def __init__(self):
        if CustomTinyImageNetDataset_test.tiny_imagenet_dataset is None:
            raise RuntimeError("Dataset not loaded. Call load_dataset() before creating instances of this class.")

    def __getitem__(self, index):
        # data_point, label = CustomTinyImageNetDataset_test.targets[index]
        data_point, _ = CustomTinyImageNetDataset_test.tiny_imagenet_dataset[index]
        label = CustomTinyImageNetDataset_test.targets[index]
        return index, (data_point, label)

    def __len__(self):
        return len(CustomTinyImageNetDataset_test.tiny_imagenet_dataset)

def parse_val_annotations(root):
    annotation_file = os.path.join(root, "val", "val_annotations.txt")
    image_to_label = {}
    with open(annotation_file, "r") as f:
        for line in f.readlines():
            parts = line.strip().split("\t")
            image_to_label[parts[0]] = parts[1]
    return image_to_label



def get_image_label(index, dataset):
    if index < 0 or index >= len(dataset):
        raise ValueError("Invalid index: must be between 0 and {} (inclusive).".format(len(dataset) - 1))

    _, (_, label) = dataset[index]
    return label



class TinyImageNet(object):
    def __init__(self, batch_size, use_gpu, num_workers, is_filter, is_mini, target_train, unlabeled_ind_train=None,
                 labeled_ind_train=None, invalidList=None):
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(64, padding=4, padding_mode='reflect'),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        pin_memory = True if use_gpu else False

        if invalidList is not None:
            labeled_ind_train = labeled_ind_train + invalidList


        trainset = CustomTinyImageNetDataset_train(transform=transform, target_train=target_train)

        if unlabeled_ind_train is None and labeled_ind_train is None:
            if is_mini:
                labeled_ind_train, unlabeled_ind_train = self.filter_known_unknown_10percent(trainset)
                self.labeled_ind_train, self.unlabeled_ind_train = labeled_ind_train, unlabeled_ind_train
            else:
                labeled_ind_train = self.filter_known_unknown(trainset)
                self.labeled_ind_train = labeled_ind_train
        else:
            self.labeled_ind_train, self.unlabeled_ind_train = labeled_ind_train, unlabeled_ind_train

        if is_filter:
            print("openset here!")
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory,
                sampler=SubsetRandomSampler(labeled_ind_train),
            )
            unlabeledloader = torch.utils.data.DataLoader(
                trainset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory,
                sampler=SubsetRandomSampler(unlabeled_ind_train),
            )
        else:
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=pin_memory,
            )

        CustomTinyImageNetDataset_test.load_dataset(transform=transform)
        testset = CustomTinyImageNetDataset_test()

        filter_ind_test = self.filter_known_unknown(testset)
        self.filter_ind_test = filter_ind_test

        if is_filter:
            testloader = torch.utils.data.DataLoader(
                testset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory,
                sampler=SubsetRandomSampler(filter_ind_test),
            )
        else:
            testloader = torch.utils.data.DataLoader(
                testset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory,
            )

        self.trainloader = trainloader
        if is_filter: self.unlabeledloader = unlabeledloader
        self.testloader = testloader
        self.num_classes = known_class

    def filter_known_unknown(self, dataset):
        filter_ind = []
        for i in range(len(dataset.targets)):
            c = dataset.targets[i]
            if c < known_class:
                filter_ind.append(i)
        return filter_ind

    def filter_known_unknown_10percent(self, dataset):
        filter_ind = []
        unlabeled_ind = []
        for i in range(len(dataset.targets)):
            c = dataset.targets[i]
            if c < known_class:
                filter_ind.append(i)
            else:
                unlabeled_ind.append(i)

        random.shuffle(filter_ind)
        labeled_ind = filter_ind[:len(filter_ind) * init_percent // 100]
        unlabeled_ind = unlabeled_ind + filter_ind[len(filter_ind) * init_percent // 100:]
        return labeled_ind, unlabeled_ind



__factory = {
    'Tiny-Imagenet': TinyImageNet,
    'cifar100': CIFAR100,
    'cifar10': CIFAR10,
    'mnist': MNIST
}


def create(name, known_class_, init_percent_, batch_size, use_gpu, num_workers, is_filter, is_mini, SEED,
           unlabeled_ind_train=None, labeled_ind_train=None, invalidList=None):
    global known_class, init_percent
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    known_class = known_class_
    init_percent = init_percent_
    
    with open('target_train.pkl', 'rb') as f:
        target_train = pickle.load(f)


    if name not in __factory.keys():
        raise KeyError("Unknown dataset: {}".format(name))
    
    if name == 'Tiny-Imagenet':
        
        return __factory[name](batch_size, use_gpu, num_workers, is_filter, is_mini, target_train , unlabeled_ind_train, labeled_ind_train,
                           invalidList)
    else:

        return __factory[name](batch_size, use_gpu, num_workers, is_filter, is_mini, unlabeled_ind_train, labeled_ind_train, invalidList)