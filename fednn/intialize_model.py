# Interface between models and the clients
# Include intialization, training for one iteration and test function
##############
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))
##############
from fednn.mnist_lenet import mnist_lenet
from fednn.cifar10cnn import cifar_cnn_3conv
from fednn.cifar100mobilenet import mobilenet
from fednn.resnet import resnet18
import torch.optim as optim

# following import is used for tesing the function of this part, they can be deleted if you delete the main() funciton
from options import args_parser
import torch
import torchvision
import torchvision.transforms as transforms
from os.path import dirname, abspath, join
from torch.autograd import Variable
from tqdm import tqdm
import torch.nn as nn


class Local_Model(object):
    def __init__(self, nn_layers, lr_method, learning_rate, lr_decay, lr_decay_epoch, momentum, weight_decay):
        self.nn_layers = nn_layers
        self.lr_method = lr_method
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.lr_decay_epoch = lr_decay_epoch
        self.momentum = momentum
        self.weight_decay = weight_decay
    #   construct the parameter
        param_dict = [{"params": self.nn_layers.parameters()}]
        self.optimizer = optim.SGD(params = param_dict,
                                  lr = learning_rate,
                                  momentum = momentum,
                                  weight_decay=weight_decay)
        self.optimizer_state_dict = self.optimizer.state_dict()
        self.criterion = nn.CrossEntropyLoss()


    def lr_scheduler(self,epoch):
        self.exp_lr_sheduler(epoch=epoch) if self.lr_method == 'exp' else self.step_lr_scheduler(epoch=epoch)
        return None

    def exp_lr_sheduler(self, epoch):
        """"""
        if (epoch + 1) % self.lr_decay_epoch:
            return None
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= self.lr_decay
            return None

    def step_lr_scheduler(self, epoch):
        # CIFAR-100 with MobileNet
        if epoch < 60 : #150
            self.learning_rate = 0.1
            for param_group in self.optimizer.param_groups:
                # print(f'epoch{epoch}')
                param_group['lr'] = 0.1
        elif epoch >= 60 and epoch < 120: #150 ~ 250
            self.learning_rate = 0.02
            for param_group in self.optimizer.param_groups:
                # print(f'epoch{epoch}')
                param_group['lr'] = 0.02 #0.01
        elif epoch >= 120 and epoch < 160:
            self.learning_rate = 0.004
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = 0.004
        elif epoch >= 160: #250
            self.learning_rate = 0.0008
            for param_group in self.optimizer.param_groups:
                # print(f'epoch{epoch}')
                param_group['lr'] = 0.0008 #0.001

    def print_current_lr(self):
        for param_group in self.optimizer.param_groups:
            print(param_group['lr'])
        return param_group['lr']

    def optimize_model(self, input_batch, label_batch):
        self.nn_layers.train(True)
        output_batch = self.nn_layers(input_batch)
        self.optimizer.zero_grad()
        batch_loss = self.criterion(output_batch, label_batch)
        batch_loss.backward()
        self.optimizer.step()
        # self.optimizer_state_dict = self.optimizer.state_dict()
        return batch_loss.item()

    def test_model(self, input_batch):
        self.nn_layers.train(False)
        with torch.no_grad():
            output_batch = self.nn_layers(input_batch)
        self.nn_layers.train(True)
        return output_batch

    def update_model(self, new_shared_layers):
        # should update state_dict of the shared_layers and the optimizer
        self.nn_layers.load_state_dict(new_shared_layers)


def initialize_model(args, device):

    if args.dataset == 'cifar10':
        if args.model == 'cnn_complex':
            nn_layers = cifar_cnn_3conv(input_channels=3, output_channels=10)
        else:
            raise ValueError('Model not implemented for CIFAR-10')

    elif args.dataset == 'mnist':
        if args.model == 'lenet':
           nn_layers = mnist_lenet(input_channels=1, output_channels=10)

        else:
            raise ValueError('Model not implemented for MNIST')

    elif args.dataset == 'cifar100':
        assert args.output_channels == 100
        assert args.iid == -1
        if args.model == 'mobilenet':
            nn_layers = mobilenet()
        elif args.model == 'resnet18':
            nn_layers = resnet18()
        else:
            raise ValueError('Model not implemented for CIFAR-100')
    else:
        raise ValueError('The dataset is not implemented for mtl yet')
    if args.cuda:
        nn_layers = nn_layers.cuda(device)
    else: raise ValueError('Wrong input for the --mtl_model and --global_model, only one is valid')

    model = Local_Model(nn_layers=nn_layers,
                        lr_method=args.lr_method,
                        learning_rate= args.lr,
                        lr_decay= args.lr_decay,
                        lr_decay_epoch= args.lr_decay_epoch,
                        momentum= args.momentum,
                        weight_decay = args.weight_decay)
    return model

def main():
    """
    For test this part
    --dataset: cifar-10
    --model: cnn_tutorial
    --lr  = 0.001
    --momentum = 0.9
    cpu only!
    check(14th/July/2019)
    :return:
    """
#     args = args_parser()
#     device = 'cpu'
#     # build dataset for testing
#     model = initialize_model(args, device)
#     # build dataset for tesing function
#     transform = transforms.Compose(
#         [transforms.ToTensor(),
#          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#     transform_train = transform
#     transform_test = transform
# #     transform_train = transforms.Compose([
# #         transforms.RandomCrop(32, padding=4),
# #         transforms.RandomHorizontalFlip(),
# #         transforms.ToTensor(),
# #         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# #     ])
# #     transform_test = transforms.Compose([
# #         transforms.ToTensor(),
# #         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# #     ])
#     parent_dir = dirname(dirname(abspath(__file__)))
#     data_path = join(parent_dir, 'data', 'cifar10')
#     trainset = torchvision.datasets.CIFAR10(root=data_path, train=True,
#                                             download=True, transform=transform_train)
#     trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
#                                               shuffle=True, num_workers=0)
#
#     testset = torchvision.datasets.CIFAR10(root=data_path, train=False,
#                                            download=True, transform=transform_test)
#     testloader = torch.utils.data.DataLoader(testset, batch_size=4,
#                                              shuffle=False, num_workers=0)
#     for epoch in tqdm(range(2)):  # loop over the dataset multiple times
#         model.exp_lr_sheduler(epoch)
#         model.print_current_lr()
#         running_loss = 0.0
#         for i, data in enumerate(trainloader, 0):
#             # get the inputs; data is a list of [inputs, labels]
#             inputs, labels = data
#             inputs = Variable(inputs).to(device)
#             labels = Variable(labels).to(device)
#             loss = model.optimize_model(input_batch= inputs,
#                                         label_batch= labels)
#
#             # print statistics
#             running_loss += loss
#             if i % 2000 == 1999:  # print every 2000 mini-batches
#                 print('[%d, %5d] loss: %.3f' %
#                       (epoch + 1, i + 1, running_loss / 2000))
#                 running_loss = 0.0
#     #             print the current learning rate
#     #             model.print_current_lr()
#
#     print('Finished Training')
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for data in testloader:
#             images, labels = data
#             outputs = model.test_model(input_batch=images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#
#     print('Accuracy of the network on the 10000 test images: %d %%' % (
#             100 * correct / total))
    # mean and std of cifar100 dataset
    CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    args = args_parser()
    args.dataset = 'cifar100'
    args.model = 'resnet18'
    args.iid = -1
    args.output_channels = 100
    args.momentum = 0.9
    args.weight_decay = 5e-4
    device = torch.device(f'cuda:{0}')
    # build dataset for testing
    model = initialize_model(args, device)
    # build dataset for tesing function
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
#     transform_train = transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     ])
#     transform_test = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     ])
    parent_dir = dirname(dirname(abspath(__file__)))
    data_path = join(parent_dir, 'data', 'cifar100')
    trainset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True,
                                                      transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=4)
    testset = torchvision.datasets.CIFAR100(root=data_path, train=False, download=True, transform=transform_test)

    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                             shuffle=False, num_workers=4)
    for epoch in tqdm(range(200)):  # loop over the dataset multiple times
        model.step_lr_scheduler(epoch)
        # model.print_current_lr()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = Variable(inputs).to(device)
            labels = Variable(labels).to(device)
            loss = model.optimize_model(input_batch= inputs,
                                        label_batch= labels)

            # print statistics
            running_loss += loss
            if i % 200 == 199:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0
    #             print the current learning rate
    #             model.print_current_lr()

    print('Finished Training')
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = Variable(images).to(device)
            labels = Variable(labels).to(device)
            outputs = model.test_model(input_batch=images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))
if __name__ == '__main__':
    main()