# The structure of the server
# The server should include the following functions:
# 1. Server initialization
# 2. Server reveives updates from the user
# 3. Server send the aggregated information back to clients
import copy
from utils.average_weights import average_weights
import torch

class Cloud():

    def __init__(self, shared_layers):
        self.receiver_buffer = {}
        self.shared_state_dict = {}
        self.id_registration = []
        self.sample_registration = {}
        self.model = shared_layers
        self.update_state_dict = shared_layers.state_dict()
        self.clock = []

    def refresh_cloudserver(self):
        self.receiver_buffer.clear()
        del self.id_registration[:]
        self.sample_registration.clear()
        return None

    def edge_register(self, edge):
        self.id_registration.append(edge.id)
        self.sample_registration[edge.id] = edge.all_trainsample_num
        return None

    def receive_from_edge(self, edge_id, eshared_state_dict):
        self.receiver_buffer[edge_id] = eshared_state_dict
        return None

    def aggregate(self, args):
        """
        I think the problem may lie in this...
        :param args:
        :return:
        """
        # print('Average Old')
        # first make the state_dict and sample into num
        # The following code may cause some problem? I am not sure whether values keeps the values int the original order
        # But when the data sample number is the same,  this is not a problem
        received_dict = [dict for dict in self.receiver_buffer.values()]
        if args.edge_average_uniform:
            sample_num = [1]*args.num_edges
        else:
            sample_num = [snum for snum in self.sample_registration.values()]
        self.update_state_dict = average_weights(w=received_dict,
                                                 s_num=sample_num)
        sd = self.model.state_dict()
        for key in sd.keys():
            sd[key] = torch.add(self.model.state_dict()[key], self.update_state_dict[key])
        self.model.load_state_dict(sd)
        # print('cloud after update')
        # print(self.model.state_dict()['stem.0.conv.weight'])
        # exit()
        return None

    def send_to_edge(self, edge):
        edge.receive_from_cloudserver(copy.deepcopy(self.model.state_dict()))
        return None

