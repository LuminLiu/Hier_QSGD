# Flow of the algorithm
# Client update(t_1) -> Edge Aggregate(t_2) -> Cloud Aggregate(t_3)

from tensorboardX import SummaryWriter
import torch
import copy
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import math

from client import Client
from edge import Edge
from cloud import Cloud
from options import args_parser
from datasets.get_data import get_dataset, show_distribution
from fednn.cifar10cnn import cifar_cnn_3conv
from fednn.mnist_lenet import mnist_lenet
from fednn.resnet import resnet18
from fednn.cifar100mobilenet import mobilenet


def fast_all_clients_test(v_test_loader, global_nn, device):
    correct_all = 0.0
    total_all = 0.0
    with torch.no_grad():
        for data in v_test_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = global_nn(inputs)
            _, predicts = torch.max(outputs, 1)
            total_all += labels.size(0)
            correct_all += (predicts == labels).sum().item()
    return correct_all, total_all

def fast_all_clients_train_loss(v_train_loader, global_nn, device):
    loss = 0.0
    num_itered = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data in v_train_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = global_nn(inputs)
            num_itered += 1
            loss += criterion(outputs, labels).item()
    loss = loss /num_itered
    return loss


def initialize_global_nn(args):
    if args.dataset == 'mnist':
        if args.model == 'lenet':
            global_nn = mnist_lenet(input_channels=1, output_channels=10)
        else: raise ValueError(f"Model{args.model} not implemented for mnist")
    elif args.dataset == 'cifar10':
        if args.model == 'cnn_complex':
            global_nn = cifar_cnn_3conv(input_channels=3, output_channels=10)
        else: raise ValueError(f"Model{args.model} not implemented for cifar")
    elif args.dataset == 'cifar100':
        if args.model == 'mobilenet':
            global_nn = mobilenet()
        elif args.model == 'resnet18':
            global_nn = resnet18()
    else: raise ValueError(f"Dataset {args.dataset} Not implemented")
    return global_nn

def ajust_interval(args, init_trainloss, global_trainloss, current_lr):
    new_tau1 = args.num_local_update * np.sqrt((global_trainloss/init_trainloss) * (args.lr/current_lr))
    new_tau1 = math.ceil(new_tau1)
    return new_tau1

def Hier_Local_QSGD_adaptive(args):
    #make experiments repeatable
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Initialize \tau_2 optimal
    if args.adjust:
        num_edge_aggregation_optimal = np.sqrt((args.latency_ec/args.latency_de)*\
                                               ((args.num_clients/args.num_edges - 1/args.q_de) / (1/args.q_de)))
        num_edge_aggregation_optimal = math.ceil(num_edge_aggregation_optimal)
        args.num_edge_aggregation = num_edge_aggregation_optimal

    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        cuda_to_use = torch.device(f'cuda:{args.gpu}')
    device = cuda_to_use if torch.cuda.is_available() else "cpu"
    print(f'Using device {device}')

    FILEOUT = f"adaptc{args.dataset}_c{args.num_clients}_e{args.num_edges}_trainr{args.train_ratio}" \
              f"t1-{args.num_local_update}_t2-{args.num_edge_aggregation}" \
              f"q_de-{args.q_de}_q_ec-{args.q_ec}-iid{args.iid}-a{args.alpha}"\
              f"_model_{args.model}epoch{args.num_communication}" \
              f"bs{args.batch_size}lr{args.lr}lr_decay_rate{args.lr_decay}" \
              f"lr_decay_epoch{args.lr_decay_epoch}ce-even{args.clients_per_edge_even}" \
              f"ec{args.clients_per_edge}ea_uni{args.edge_average_uniform}"
    print(FILEOUT)
    print(f'Args parser is {args}')
    writer = SummaryWriter(comment=FILEOUT)

    # Build dataloaders
    train_loaders, test_loaders, v_train_loader, v_test_loader = get_dataset(args.dataset_root, args.dataset, args)
    if args.show_dis:
        for i in range(args.num_clients):
            train_loader = train_loaders[i]
            print(len(train_loader.dataset))
            distribution = show_distribution(train_loader, args)
            print("train dataloader {} distribution".format(i))
            print(distribution)

        for i in range(args.num_clients):
            test_loader = test_loaders[i]
            test_size = len(test_loaders[i].dataset)
            print(len(test_loader.dataset))
            distribution = show_distribution(test_loader, args)
            print("test dataloader {} distribution".format(i))
            print(f"test dataloader size {test_size}")
            print(distribution)

    # initialize clients and server
    clients = []
    for i in range(args.num_clients):
        clients.append(Client(id=i,
                              train_loader=train_loaders[i],
                              test_loader=test_loaders[i],
                              args=args,
                              device=device)
                       )

    initilize_parameters = list(clients[0].model.nn_layers.parameters())
    nc = len(initilize_parameters)
    for client in clients:
        user_parameters = list(client.model.nn_layers.parameters())
        for i in range(nc):
            user_parameters[i].data[:] = initilize_parameters[i].data[:]

    # New an NN model for testing error
    global_nn = initialize_global_nn(args)
    if args.cuda:
        global_nn = global_nn.cuda(device)
    global_nn_parameters = list(global_nn.parameters())
    for i in range(nc):
        global_nn_parameters[i].data[:] = initilize_parameters[i].data[:]
    init_trainloss = fast_all_clients_train_loss(v_train_loader, global_nn, device)

    best_avg_acc = 0.0
    best_train_loss = init_trainloss

    walltime_record = 0.0
    walltime = 0.0

    # Initialize edge server and assign clients to the edge server
    # Can be extended here, how to assign the clients to the edge
    edges = []
    cids = np.arange(args.num_clients)

    if args.clients_per_edge_even :
        clients_per_edge = [int(args.num_clients / args.num_edges)] * args.num_edges
    else:
        clients_per_edge = [int(item) for item in args.clients_per_edge.split(',')]

    print(type(clients_per_edge))
    p_clients = [0.0] * args.num_edges

 # This is randomly assign the clients to edges
    for i in range(args.num_edges):
        #Randomly select clients and assign them
        np.random.seed(args.seed)
        selected_cids = np.random.choice(cids, clients_per_edge[i], replace=False)
        cids = list (set(cids) - set(selected_cids))
        edges.append(Edge(id = i,
                          cids = selected_cids,
                          shared_layers = copy.deepcopy(clients[0].model.nn_layers)))
        [edges[i].client_register(clients[cid]) for cid in selected_cids]
        edges[i].all_trainsample_num = sum(edges[i].sample_registration.values())
        p_clients[i] = [sample / float(edges[i].all_trainsample_num) for sample in
                list(edges[i].sample_registration.values())]
        edges[i].refresh_edgeserver()

    # Initialize cloud server
    cloud = Cloud(shared_layers=copy.deepcopy(clients[0].model.nn_layers))

    # First the clients report to the edge server their training samples
    [cloud.edge_register(edge=edge) for edge in edges]
    p_edge = [sample / sum(cloud.sample_registration.values()) for sample in
                list(cloud.sample_registration.values())]
    cloud.refresh_cloudserver()

    new_num_local_update = args.num_local_update
    #Begin training
    for num_comm in tqdm(range(args.num_communication)):
        cloud.refresh_cloudserver()
        [cloud.edge_register(edge=edge) for edge in edges]
        for num_edgeagg in range(args.num_edge_aggregation):
            for i,edge in enumerate(edges):
                edge.refresh_edgeserver()
                client_loss = 0.0
                selected_cnum = max(int(clients_per_edge[i] * args.frac),1)
                np.random.seed(args.seed)
                selected_cids = np.random.choice(edge.cids,
                                                 selected_cnum,
                                                 replace = False,
                                                 p = p_clients[i])
                for selected_cid in selected_cids:
                    edge.client_register(clients[selected_cid])
                for selected_cid in selected_cids:
                    edge.send_to_client(clients[selected_cid])
                    clients[selected_cid].sync_with_edgeserver()
                    client_loss += clients[selected_cid].local_update(num_iter= new_num_local_update,
                                                                      device = device)

                    clients[selected_cid].send_to_edgeserver(edge, args.q_de, False, args.q_method)

                edge.aggregate(args)

        # Now begin the cloud aggregation
        for edge in edges:
            edge.send_to_cloudserver(cloud, args.q_ec, args.q_method)
        cloud.aggregate(args)
        for edge in edges:
            cloud.send_to_edge(edge)

        # # for debugging
        # correct, total = clients[0].test_model(device)
        # acc = correct / total
        # print(f'client acc after aggregation is {acc}')

        # # for debugging
        # sd_client = clients[0].model.shared_layers.state_dict()
        # sd_edge = edges[0].model.state_dict()
        # sd_cloud = cloud.model.state_dict()
        # for key in sd_client.keys():
        #     dif_ce = torch.add(sd_client[key], -sd_edge[key])
        #     dif_cc = torch.add(sd_client[key], -sd_cloud[key])
        #     if dif_ce.sum().data > 1e-5:
        #         print(f'Key is {key}, dif client & edge')
        #     if dif_cc.sum().data > 1e-5:
        #         print(f'Key is {key}, dif client & cloud')

        # Use the virtual testloader for testing
        global_nn.load_state_dict(state_dict = copy.deepcopy(cloud.model.state_dict()))
        global_nn.train(False)
        correct_all_v, total_all_v = fast_all_clients_test(v_test_loader, global_nn, device)
        global_trainloss = fast_all_clients_train_loss(v_train_loader, global_nn, device)

        walltime_record += args.latency_comp * new_num_local_update * args.num_edge_aggregation + args.latency_de * args.num_edge_aggregation + args.latency_ec

        avg_acc_v = correct_all_v / total_all_v
        writer.add_scalar(f'All_Avg_Test_Acc_cloudagg_Vtest',
                          avg_acc_v,
                          num_comm + 1)
        writer.add_scalar(f'Glbal_TrainLoss',
                          global_trainloss,
                          num_comm+1)
        writer.add_scalar(f'walltime',
                          walltime_record,
                          num_comm+1)
        writer.add_scalar(f'tau1',
                          new_num_local_update,
                          num_comm+1)

        if avg_acc_v > best_avg_acc:
            best_avg_acc = avg_acc_v
        if global_trainloss < best_train_loss:
            best_train_loss = global_trainloss


        if args.adjust:
            walltime += args.latency_comp * new_num_local_update * args.num_edge_aggregation + args.latency_de * args.num_edge_aggregation + args.latency_ec
            current_lr = clients[0].model.print_current_lr()
            if walltime >= args.adjust_interval:
                new_num_local_update = ajust_interval(args, init_trainloss, global_trainloss, current_lr)
                print(f'new tau1{new_num_local_update}')
                walltime = 0.0


    writer.close()
    print(f"The final best virtual acc is {best_avg_acc}")
    print(f'The final best virtual train loss is {best_train_loss}')

def main():
    args = args_parser()
    args.client_per_edge = [int(item) for item in args.clients_per_edge.split(',')]
    Hier_Local_QSGD_adaptive(args)

if __name__ == '__main__':
    main()