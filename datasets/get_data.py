# Interface between the dataset and client
# For artificially partitioned dataset, params include num_clients, dataset
# For federated datasets and multitask datasets, params include dataset

import os
import numpy as np
import torch
import random

import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from options import args_parser


class DatasetSplitWithIndex(Dataset):

    def __init__(self, dataset, idxs):
        super(DatasetSplitWithIndex, self).__init__()
        self.dataset = dataset
        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        """
        :param item: index in the split dataset
        :return: sample, label and the index in the original complete dataset
        """
        image, target = self.dataset[self.idxs[item]]
        return image, target


def iid_esize_split(dataset, available_idxes, args, kwargs, is_shuffle=True):
    """
    split the dataset to users
    Return:
        dict of the data_loaders
    """
    sum_samples = len(available_idxes)
    num_samples_per_client = int(sum_samples / args.num_clients)
    # change from dict to list
    data_loaders = [0] * args.num_clients
    dict_users, all_idxs = {}, available_idxes
    # dict_users, all_idxs = {}, dataset.indices
    for i in range(args.num_clients):
        np.random.seed(args.seed)
        dict_users[i] = np.random.choice(all_idxs, num_samples_per_client, replace=False)
        all_idxs = list(set(all_idxs) - set(dict_users[i]))
        data_loaders[i] = DataLoader(DatasetSplitWithIndex(dataset, dict_users[i]),
                                     batch_size=args.batch_size,
                                     shuffle=is_shuffle, **kwargs
                                     )

    return data_loaders


def niid_esize_split(dataset, available_idxes, args, kwargs, is_shuffle=True):
    data_loaders = [0] * args.num_clients
    # each client has only two classes of the network
    num_shards = 2 * args.num_clients
    # the number of images in one shard
    num_imgs = int(len(available_idxes) / num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(args.num_clients)}
    idxs = available_idxes
    # is_shuffle is used to differentiate between train and test
    labels = [dataset.targets[idx] for idx in available_idxes]
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    # sort the data according to their label
    idxs = idxs_labels[0, :]
    idxs = idxs.astype(int)

    # divide and assign
    for i in range(args.num_clients):
        np.random.seed(args.seed)
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs: (rand + 1) * num_imgs]), axis=0)
            dict_users[i] = dict_users[i].astype(int)
        print(f'{i} dictuser {len(dict_users[i])}')
        data_loaders[i] = DataLoader(DatasetSplitWithIndex(dataset, dict_users[i]),
                                     batch_size=args.batch_size,
                                     shuffle=is_shuffle, **kwargs)
    return data_loaders


def dirichlet_split_shortage_handle(dataclass_index, sampleOfClass, assigned_cls, sampleofID_c, random_seed, num_classes):
    """
    Use self-invoking function to handle the sample shortage problem in dirichlet split
    :param dataclass_index: list of list, len = 10, each element is the list of index of each of the 10 clses
    :param sampleOfClass: int, is the number of data samples to be assigned for the usr for the assigned_cls
    :param assigned_cls: the current cls being assigned
    :param sampleofID_c: the list of the sample of ID for current client
    :param random_seed: seed
    :return select_ID : the selected indexes for assigned cls
    :return dataclass_index: the list of the remaining available idxes
    """
    np.random.seed(random_seed)
    if len(dataclass_index[assigned_cls]) >= sampleOfClass:
        select_ID = random.sample(dataclass_index[assigned_cls],
                                  sampleOfClass)
        dataclass_index[assigned_cls] = list(set(dataclass_index[assigned_cls]) - set(select_ID))
        sampleofID_c += select_ID
    else:
        shortage = sampleOfClass - len(dataclass_index[assigned_cls])
        select_ID = random.sample(dataclass_index[assigned_cls], len(dataclass_index[assigned_cls]))
        dataclass_index[assigned_cls] = list(set(dataclass_index[assigned_cls]) - set(select_ID))
        sampleofID_c += select_ID
        dataclass_num = [len(dataclass_index[cls]) for cls in range(num_classes)]
        max_cls = np.argmax(dataclass_num)
        sampleofID_c, dataclass_index = dirichlet_split_shortage_handle(dataclass_index=dataclass_index,
                                                                        sampleOfClass=shortage,
                                                                        assigned_cls=max_cls,
                                                                        sampleofID_c=sampleofID_c,
                                                                        random_seed=random_seed)
    return sampleofID_c, dataclass_index


def dirichlet_split(dataset, available_idxes, args, kwargs, is_shuffle):
    data_loaders = [0] * args.num_clients
    # equal division in data sample number
    datanumber = int(len(available_idxes) / args.num_clients)
    # print(f'initial data sample number {datanumber}')
    dataclass_index = [[] for i in range(args.output_channels)]
    idxs = available_idxes
    labels = [dataset.targets[idx] for idx in available_idxes]
    for idx in available_idxes:
        dataclass_index[dataset.targets[idx]].append(idx)

    # for cls in range(10):
    #     print(f'Length of the class{cls} is {len(dataclass_index[cls])}')
    np.random.seed(args.seed)
    dirichlet_label = np.random.dirichlet([args.alpha] * args.output_channels,
                                          args.num_clients)
    # dirichlet_label: size (num_class * num_clients) matrix, rsum = 1
    if args.double_stochastic:
        dirichlet_label = make_double_stochstic(dirichlet_label)

    sampleOfID = [[] for _ in range(args.num_clients)]
    for client in range(args.num_clients):
        # np.random.seed(args.seed)
        probs = dirichlet_label[client]
        sampleOfClass = [int(datanumber * prob) for prob in probs]
        for i in range(args.output_channels):
            sampleOfID[client], dataclass_index = dirichlet_split_shortage_handle(dataclass_index=dataclass_index,
                                                                                  sampleOfClass=sampleOfClass[i],
                                                                                  assigned_cls=i,
                                                                                  sampleofID_c=sampleOfID[client],
                                                                                  random_seed=args.seed,
                                                                                  num_classes = args.output_channels)
            # np.random.seed(args.seed)
            # if len(dataclass_index[i]) >= sampleOfClass[i]:
            #     select_ID = random.sample(dataclass_index[i],
            #                               sampleOfClass[i])
            #     sampleOfID[client] += select_ID
            #     dataclass_index[i] = list(set(dataclass_index[i])-set(select_ID))
            #     max_cls = np.argmax(dataclass_num)
            #
            # else:
            #     shortage = (sampleOfClass[i] - len(dataclass_index[i]))
            #     select_ID = random.sample(dataclass_index[i],
            #                                  len(dataclass_index[i]))
            #     sampleOfID[client]+= select_ID
            #     dataclass_index[i] = list(set(dataclass_index[i]) - set(select_ID))
            #     # dynamically adjusting the max_cls
            #     dataclass_num = [len(dataclass_index[cls]) for cls in range(10)]
            #     max_cls = np.argmax(dataclass_num)
            #     #borrow from the max_cls
            #     np.random.seed(args.seed)
            #     try:
            #         select_ID = random.sample(dataclass_index[max_cls], shortage)
            #     except:
            #         print(f'assigning for client{client}')
            #         print(f'shortage for cls {i} is {shortage}')
            #         print(f'Now using max class {max_cls} to compensate')
            #         print(f'max class length {dataclass_num[max_cls]}')
            #         raise ValueError('Dirichlet distributed data assigning error')
            #     sampleOfID[client] += select_ID
            #     dataclass_index[max_cls] = list(set(dataclass_index[max_cls]) - set(select_ID))
        data_loaders[client] = DataLoader(DatasetSplitWithIndex(dataset, sampleOfID[client]),
                                          batch_size=args.batch_size,
                                          shuffle=is_shuffle, **kwargs)
    return data_loaders


def make_double_stochstic(x):
    # rsum = 0, which is a inherent property of the dirichlet matrix
    # here, its kinda like to rescale each row, so that the csum are equal for all columns
    # But I do not quite understand how it works
    # It is like a puzzle
    rsum = None
    csum = None

    n = 0
    while n < 1000 and (np.any(rsum != 1) or np.any(csum != 1)):
        x /= x.sum(0)
        x = x / x.sum(1)[:, np.newaxis]
        rsum = x.sum(1)
        csum = x.sum(0)
        n += 1

    # x = x / x.sum(axis=0).reshape(1,-1)
    return x


def split_data(dataset, args, available_idxes, kwargs, is_shuffle=True):
    """
    return dataloaders
    """
    if is_shuffle:  # only train set are split in a non-iid way
        if args.iid == 1:
            data_loaders = iid_esize_split(dataset, available_idxes, args, kwargs, is_shuffle)
        elif args.iid == 0:
            data_loaders = niid_esize_split(dataset, available_idxes, args, kwargs, is_shuffle)
        elif args.iid == -1:
            data_loaders = dirichlet_split(dataset, available_idxes, args, kwargs, is_shuffle)
        else:
            raise ValueError('Data Distribution pattern `{}` not implemented '.format(args.iid))
    else:
        data_loaders = iid_esize_split(dataset, available_idxes, args, kwargs, is_shuffle)
    return data_loaders


def get_mnist(dataset_root, args):
    is_cuda = args.cuda
    kwargs = {'num_workers': 4, 'pin_memory': True} if is_cuda else {}
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train = datasets.MNIST(os.path.join(dataset_root, 'mnist'), train=True,
                           download=True, transform=transform)
    test = datasets.MNIST(os.path.join(dataset_root, 'mnist'), train=False,
                          download=True, transform=transform)

    all_trainindxes = [i for i in range(len(train))]

    train_loaders = split_data(dataset=train,
                               args=args,
                               available_idxes=all_trainindxes,
                               kwargs=kwargs,
                               is_shuffle=True)

    test_loaders = split_data(dataset=test,
                              args=args,
                              available_idxes=[i for i in range(len(test))],
                              kwargs=kwargs,
                              is_shuffle=False)

    v_train_loader = DataLoader(DatasetSplitWithIndex(train, [i for i in range(len(train))]),
                                batch_size=args.batch_size * args.num_clients,
                                shuffle=True, **kwargs)

    v_test_loader = DataLoader(DatasetSplitWithIndex(test, [i for i in range(len(test))]),
                               batch_size=args.batch_size * args.num_clients,
                               shuffle=False, **kwargs)

    return train_loaders, test_loaders, v_train_loader, v_test_loader


def get_cifar10(dataset_root, args):
    is_cuda = args.cuda
    kwargs = {'num_workers': 0, 'pin_memory': True} if is_cuda else {}
    # the following transform may be suitable for resnet
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train = datasets.CIFAR10(os.path.join(dataset_root, 'cifar10'), train=True,
                             download=True, transform=transform_train)
    test = datasets.CIFAR10(os.path.join(dataset_root, 'cifar10'), train=False,
                            download=True, transform=transform_test)

    all_trainindxes = [i for i in range(len(train))]

    train_loaders = split_data(dataset=train,
                               args=args,
                               available_idxes=all_trainindxes,
                               kwargs=kwargs,
                               is_shuffle=True)

    test_loaders = split_data(dataset=test,
                              args=args,
                              available_idxes=[i for i in range(len(test))],
                              kwargs=kwargs,
                              is_shuffle=False)

    v_train_loader = DataLoader(DatasetSplitWithIndex(train, [i for i in range(len(train))]),
                                batch_size=args.batch_size * args.num_clients,
                                shuffle=True, **kwargs)

    v_test_loader = DataLoader(DatasetSplitWithIndex(test, [i for i in range(len(test))]),
                               batch_size=args.batch_size * args.num_clients,
                               shuffle=False, **kwargs)

    return train_loaders, test_loaders, v_train_loader, v_test_loader

def get_cifar100(dataset_root, args):
    # only dirichlet non-iid is implemented for cifar-10
    is_cuda = args.cuda
    kwargs = {'num_workers': 0, 'pin_memory': True} if is_cuda else {}
    CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    transform_train = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
    ])

    train = datasets.CIFAR100(os.path.join(dataset_root, 'cifar100'), train=True, download=True,
                                             transform=transform_train)
    test = datasets.CIFAR100(os.path.join(dataset_root, 'cifar100'), train=False, download=True,
                              transform=transform_test)

    all_trainindxes = [i for i in range(len(train))]

    train_loaders = split_data(dataset=train,
                               args=args,
                               available_idxes=all_trainindxes,
                               kwargs=kwargs,
                               is_shuffle=True)

    test_loaders = split_data(dataset=test,
                              args=args,
                              available_idxes=[i for i in range(len(test))],
                              kwargs=kwargs,
                              is_shuffle=False)

    v_train_loader = DataLoader(DatasetSplitWithIndex(train, [i for i in range(len(train))]),
                                batch_size=args.batch_size * args.num_clients,
                                shuffle=True, **kwargs)

    v_test_loader = DataLoader(DatasetSplitWithIndex(test, [i for i in range(len(test))]),
                               batch_size=args.batch_size * args.num_clients,
                               shuffle=False, **kwargs)

    return train_loaders, test_loaders, v_train_loader, v_test_loader


def get_dataset(dataset_root, dataset, args):
    if dataset == 'mnist':
        train_loaders, test_loaders, v_train_loader, v_test_loader = get_mnist(dataset_root, args)
    elif dataset == 'cifar10':
        train_loaders, test_loaders, v_train_loader, v_test_loader = get_cifar10(dataset_root,
                                                                                 args)
    elif dataset =='cifar100':
        train_loaders, test_loaders, v_train_loader, v_test_loader = get_cifar100(dataset_root,args)
    else:
        raise ValueError('Dataset `{}` not found'.format(dataset))
    return train_loaders, test_loaders, v_train_loader, v_test_loader


def show_distribution(dataloader, args):
    """
    show the distribution of the data on certain client with dataloader
    return:
        percentage of each class of the label
    """
    if args.dataset == 'mnist':
        labels = dataloader.dataset.dataset.targets.numpy()
    elif args.dataset == 'cifar10':
        try:
            labels = dataloader.dataset.dataset.targets
        except:
            labels = dataloader.dataset.targets
    elif args.dataset == 'cifar100':
        try:
            labels = dataloader.dataset.dataset.targets
        except:
            labels = dataloader.dataset.targets
    else:
        raise ValueError("`{}` dataset not included".format(args.dataset))
    num_samples = len(dataloader.dataset)
    idxs = [i for i in range(num_samples)]
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    distribution = [0] * len(unique_labels)
    if type(dataloader.dataset) == DatasetSplitWithIndex:
        for idx in idxs:
            img, label = dataloader.dataset[idx]
            distribution[label] += 1
    elif type(dataloader.dataset) == datasets.cifar.CIFAR10:
        for idx in idxs:
            img, label = dataloader.dataset[idx]
            distribution[label] += 1
    elif type(dataloader.dataset) == datasets.mnist.MNIST:
        for idx in idxs:
            img, label = dataloader.dataset[idx]
            distribution[label] += 1
    elif type(dataloader.dataset) == datasets.cifar.CIFAR100:
        for idx in idxs:
            img, label = dataloader.dataset[idx]
            distribution[label] += 1
    else:
        raise ValueError(f'Dataset type wrong!')
    distribution = np.array(distribution)
    distribution = distribution / num_samples
    return distribution


if __name__ == '__main__':
    args = args_parser()
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    train_loaders, test_loaders, v_shared_loader, v_test_loader = get_dataset(args.dataset_root, args.dataset, args)
    print(f"The dataset is {args.dataset} divided into {args.num_clients} clients/tasks in an iid = {args.iid} way")
    print(f'public ratio is {args.public_ratio}')

    for i in range(args.num_clients):
        train_loader = train_loaders[i]
        print(len(train_loader.dataset))
        # print(type(train_loader.dataset))
        distribution = show_distribution(train_loader, args)
        print("dataloader {} distribution".format(i))
        print(distribution)

    # if shared_train_set!=None:
    #     print(len(shared_train_set))
    #     shared_train_loader = DataLoader(shared_train_set,
    #                                      batch_size=args.bs_d,
    #                                      shuffle=True)
    # distribution = show_distribution(shared_train_loader, args)
    # print("shared_trainloader distribution".format(i))
    # print(distribution)
    # here is an error
    # train_set = shared_train_set.dataset

    # shared_idx = shared_train_set.idxs
    # len_shared = shared_train_set.__len__()
    # select_shared_idx = np.random.choice(shared_idx, 100, replace=False)
    # select_shared_dataloader = DataLoader(DatasetSplitWithIndex(shared_train_set.dataset, select_shared_idx),
    #                                       batch_size=args.bs_d,
    #                                       shuffle=True)
    # print(len(select_shared_dataloader.dataset))
    # for _, data in enumerate(select_shared_dataloader, 0):
    #     inputs, _, idxes = data

    # distribution = show_distribution(select_shared_dataloader, args)
    # print("select shared_trainloader distribution")
    # print(select_shared_dataloader.dataset.__len__())
    # print(distribution)

    print(len(v_test_loader.dataset))
    # print(type(v_test_loader.dataset))
    distribution = show_distribution(v_test_loader, args)
    print("vtestloader distribution")
    print(distribution)

    # for i in range(args.num_clients):
    #     test_loader = test_loaders[i]
    #     print(len(train_loader.dataset))
    #     # print(type(train_loader.dataset))
    #     distribution = show_distribution(test_loader, args)
    #     print("dataloader {} distribution".format(i))
    #     print(distribution)


