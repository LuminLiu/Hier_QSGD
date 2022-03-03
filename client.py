# The structure of the client
# Should include following funcitons
# 1. Client intialization, dataloaders, model(include optimizer)
# 2. Client model update
# 3. Client send the quantized updates to server
# 4. Client receives new model from server
# 5. Client modify local model based on the feedback from the server
from torch.autograd import Variable
import torch
from fednn.intialize_model import initialize_model
import copy
from utils.quantization import quantization_nn


class Client():

    def __init__(self, id, train_loader, test_loader, args, device):
        self.id = id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = initialize_model(args, device)
        self.q_update = copy.deepcopy(self.model)
        self.receiver_buffer = {}
        self.batch_size = args.batch_size
        #record local update epoch
        self.lr_method = args.lr_method
        self.num_batches = 0
        self.epoch = 0
        self.epoch_th = len(self.train_loader)

    def local_update(self, num_iter, device):
        itered_num = 0
        loss = 0.0
        end = False
        # the upperbound selected in the following is because it is expected that one local update will never reach 1000
        for epoch in range(1000):
            for data in self.train_loader:
                inputs, labels = data
                inputs = Variable(inputs).to(device)
                labels = Variable(labels).to(device)
                loss += self.model.optimize_model(input_batch=inputs,
                                                  label_batch=labels)
                itered_num += 1
                self.num_batches +=1
                if itered_num >= num_iter:
                    end = True
                    break
            if self.num_batches >= self.epoch_th:
                self.epoch += int(self.num_batches/self.epoch_th)
                self.model.lr_scheduler(epoch = self.epoch)
                self.num_batches = self.num_batches % self.epoch_th
            if end: break
        loss /= num_iter
        return loss

    def test_model(self, device):
        correct = 0.0
        total = 0.0
        with torch.no_grad():
            for data in self.test_loader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = self.model.test_model(input_batch= inputs)
                _, predict = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predict == labels).sum().item()
        return correct, total

    def send_to_edgeserver(self, edgeserver, compression_ratio, compute_real_q, q_method):
        self.compute_update(initial_model= edgeserver.model, updated_model= self.model)
        # Now we decomment the random sparsification first
        real_q = 0.0
        if compute_real_q:
            fp_update = copy.deepcopy(self.q_update)
        self.q_update = quantization_nn(self.q_update, compression_ratio, q_method)
        if compute_real_q:
            real_q = self.compute_real_q(fp_update)
        # print('client after update')
        # print(self.model.nn_layers.state_dict()['stem.0.conv.weight'])
        edgeserver.receive_from_client(client_id= self.id,
                                        cshared_state_dict = copy.deepcopy(self.q_update.nn_layers.state_dict())
                                        )
        return real_q

    def receive_from_edgeserver(self, shared_state_dict):
        self.receiver_buffer = shared_state_dict
        return None

    def sync_with_edgeserver(self):
        """
        The global has already been stored in the buffer
        :return: None
        """
        # self.model.shared_layers.load_state_dict(self.receiver_buffer)
        self.model.update_model(self.receiver_buffer)
        return None

    def compute_update(self, initial_model, updated_model):
        initial_param = list(initial_model.parameters())
        updated_param = list(updated_model.nn_layers.parameters())
        gradient_param = list(self.q_update.nn_layers.parameters())
        nc = len(initial_param)
        for i in range(nc):
            gradient_param[i].data[:] = updated_param[i].data[:] - initial_param[i].data[:]
        return None

    def compute_real_q(self, fp_update):
        fp_total_norm = 0.0
        variance_norm = 0.0
        fp_param = list(fp_update.nn_layers.parameters())
        q_param = list(self.q_update.nn_layers.parameters())
        nc = len(fp_param)
        for i in range(nc):
            fp_total_norm += fp_param[i].data[:].norm().item() **2
            variance_norm += (fp_param[i].data[:] - q_param[i].data[:]).norm().item() **2
        real_q = variance_norm / fp_total_norm
        return real_q


