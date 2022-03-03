# Flow of the algorithm
# Client update(t_1) -> Edge Aggregate(t_2) -> Cloud Aggregate(t_3)

from tensorboardX import SummaryWriter
import torch
import copy
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.autograd import Variable

from client import Client
from edge import Edge
from cloud import Cloud
from options import args_parser
from datasets.get_data import get_dataset, show_distribution
from fednn.cifar10cnn import cifar_cnn_3conv
from fednn.mnist_lenet import mnist_lenet
from fednn.resnet import resnet18
from fednn.cifar100mobilenet import mobilenet



def validate_state_dicts(model_state_dict_1, model_state_dict_2):
    if len(model_state_dict_1) != len(model_state_dict_2):
        print(
            f"Length mismatch: {len(model_state_dict_1)}, {len(model_state_dict_2)}"
        )
        return False

    # Replicate modules have "module" attached to their keys, so strip these off when comparing to local model.
    if next(iter(model_state_dict_1.keys())).startswith("module"):
        model_state_dict_1 = {
            k[len("module") + 1 :]: v for k, v in model_state_dict_1.items()
        }

    if next(iter(model_state_dict_2.keys())).startswith("module"):
        model_state_dict_2 = {
            k[len("module") + 1 :]: v for k, v in model_state_dict_2.items()
        }

    for ((k_1, v_1), (k_2, v_2)) in zip(
        model_state_dict_1.items(), model_state_dict_2.items()
    ):
        if k_1 != k_2:
            print(f"Key mismatch: {k_1} vs {k_2}")
            return False
        # convert both to the same CUDA device
        if str(v_1.device) != "cuda:0":
            v_1 = v_1.to("cuda:0" if torch.cuda.is_available() else "cpu")
        if str(v_2.device) != "cuda:0":
            v_2 = v_2.to("cuda:0" if torch.cuda.is_available() else "cpu")

        if not torch.allclose(v_1, v_2):
            print(f"Tensor mismatch: {v_1} vs {v_2}")
            print(f'mismatch key{k_1}')
            return False


def fast_all_clients_test(v_test_loader, global_nn, device):
    correct_all = 0.0
    total_all = 0.0
    with torch.no_grad():
        for data in v_test_loader:
            inputs, labels = data
            inputs = Variable(inputs).to(device)
            labels = Variable(labels).to(device)
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
            inputs = Variable(inputs).to(device)
            labels = Variable(labels).to(device)
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

def Hier_Local_QSGD(args):
    #make experiments repeatable
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        cuda_to_use = torch.device(f'cuda:{args.gpu}')
    device = cuda_to_use if torch.cuda.is_available() else "cpu"
    print(f'Using device {device}')
    FILEOUT = f"{args.dataset}_c{args.num_clients}_e{args.num_edges}_trainr{args.train_ratio}" \
              f"t1-{args.num_local_update}_t2-{args.num_edge_aggregation}" \
              f"q_de-{args.q_de}_q_ec-{args.q_ec}-q_m{args.q_method}-iid{args.iid}-a{args.alpha}"\
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
        # show trainloader distribution
        for i in range(args.num_clients):
            train_loader = train_loaders[i]
            print(len(train_loader.dataset))
            distribution = show_distribution(train_loader, args)
            print("train dataloader {} distribution".format(i))
            print(distribution)
        # # show testloader distribution
        # for i in range(args.num_clients):
        #     test_loader = test_loaders[i]
        #     test_size = len(test_loaders[i].dataset)
        #     print(len(test_loader.dataset))
        #     distribution = show_distribution(test_loader, args)
        #     print("test dataloader {} distribution".format(i))
        #     print(f"test dataloader size {test_size}")
        #     print(distribution)
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
    # get_edge_class(args, edges, clients)
    # Initialize cloud server
    cloud = Cloud(shared_layers=copy.deepcopy(clients[0].model.nn_layers))

    # First the clients report to the edge server their training samples
    [cloud.edge_register(edge=edge) for edge in edges]
    # p_edge = [sample / sum(cloud.sample_registration.values()) for sample in
    #             list(cloud.sample_registration.values())]
    cloud.refresh_cloudserver()

    #New an NN model for testing error
    global_nn = initialize_global_nn(args)
    if args.cuda:
        global_nn = global_nn.cuda(device)

    best_avg_acc = 0.0
    best_train_loss = 100000
    real_q = 0.0
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
                    client_loss += clients[selected_cid].local_update(num_iter=args.num_local_update,
                                                                      device = device)
                    clients[selected_cid].send_to_edgeserver(edge, args.q_de, False, args.q_method)

                    # uncomment following if we need to compute real_q here
                    # # only compute the client[0]'s real_q before the cloud comm round
                    # if num_edgeagg < args.num_edge_aggregation - 1:
                    #     clients[selected_cid].send_to_edgeserver(edge, args.q_de, False, args.q_method)
                    # else:
                    #     if selected_cid:
                    #         clients[selected_cid].send_to_edgeserver(edge, args.q_de, False, args.q_method)
                    #     else:
                    #         real_q = clients[selected_cid].send_to_edgeserver(edge, args.q_de, True, args.q_method)

                # #     for the use of debugging
                # correct, total = clients[0].test_model(device)
                # acc = correct / total
                # print(f'acc before aggregation is {acc}')
                # edge_loss[i] = client_loss
                # edge_sample[i] = sum(edge.sample_registration.values())

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
        # sd_client = clients[0].model.nn_layers.state_dict()
        # sd_edge = edges[0].model.state_dict()
        # sd_cloud = cloud.model.state_dict()
        # validate_state_dicts(sd_client, sd_edge)
        # validate_state_dicts(sd_client, sd_cloud)
        # exit()

        global_nn.load_state_dict(state_dict = copy.deepcopy(cloud.model.state_dict()))
        # global_nn.load_state_dict(state_dict=copy.deepcopy(edges[0].model.state_dict()))
        # global_nn.load_state_dict(state_dict=copy.deepcopy(clients[0].model.nn_layers.state_dict()))

        global_nn.train(False)
        correct_all_v, total_all_v = fast_all_clients_test(v_test_loader, global_nn, device)
        global_trainloss = fast_all_clients_train_loss(v_train_loader, global_nn, device)
        avg_acc_v = correct_all_v / total_all_v
        writer.add_scalar(f'All_Avg_Test_Acc_cloudagg_Vtest',
                          avg_acc_v,
                          num_comm + 1)
        writer.add_scalar(f'Glbal_TrainLoss',
                          global_trainloss,
                          num_comm+1)
        writer.add_scalar(f'Real_Q1',
                          real_q,
                          num_comm + 1)
        if avg_acc_v > best_avg_acc:
            best_avg_acc = avg_acc_v
        if global_trainloss < best_train_loss:
            best_train_loss = global_trainloss
        if args.verbose:
            # correct_c, all_c = clients[0].test_model(device)
            # acc_c = correct_c / all_c
            # print(f'client test acc {acc_c} at comm round {num_comm+1}')
            print(f'epoch is {clients[0].epoch}')
            print(f'accumulated num_batches is {clients[0].num_batches}')
            print(f'epoch_th is {clients[0].epoch_th}')
            clients[0].model.print_current_lr()
            print(f'All_Avg_Test_Acc_cloudagg_Vtest{avg_acc_v} at comm round{num_comm+1}')
            print(f'Glbal_TrainLoss{global_trainloss}at comm round{num_comm+1}')

    writer.close()
    print(f"The final best virtual acc is {best_avg_acc}")
    print(f'The final best virtual train loss is {best_train_loss}')

def main():
    args = args_parser()
    args.client_per_edge = [int(item) for item in args.clients_per_edge.split(',')]
    Hier_Local_QSGD(args)

if __name__ == '__main__':
    main()