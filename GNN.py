import os
import numpy as np
import random
import time
import json

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.data as pygdat
from torch_geometric.nn import GCNConv  
from torch_geometric.nn import global_mean_pool 

import JXPExecute


def setup_seed(seed):
    torch.manual_seed(seed)          
    torch.cuda.manual_seed(seed)     
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class GNN(nn.Module):
    def __init__(self,num_layers, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.num_layers = num_layers

        self.batch_norms = torch.nn.ModuleList()
        self.linears_prediction = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()
        self.dropout = torch.nn.Dropout(p=0.5)
        self.fc1 = torch.nn.Linear(hidden_dim,output_dim)

        for layer in range(self.num_layers-1):
            if layer == 0:
                self.convs.append(GCNConv(input_dim,hidden_dim,add_self_loops=False,normalize=False))
            else:
                self.convs.append(GCNConv(hidden_dim,hidden_dim,add_self_loops=False,normalize=False))

            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # self.convs.append(GCNConv(hidden_dim,output_dim,add_self_loops=False))
        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(nn.Linear(input_dim, output_dim))
            else:
                self.linears_prediction.append(nn.Linear(hidden_dim, output_dim))

    def forward(self,data):
        # x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x, edge_index, edge_weight = data.x.float(), data.edge_index, data.edge_weight.float()

        hidden_res = [x]

        for i in range(self.num_layers-1):
            x = self.convs[i](x,edge_index,edge_weight)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            hidden_res.append(x)

        readout_all_layer = 0
        for i,x in enumerate(hidden_res):
            readout = global_mean_pool(x,data.batch)
            tmp = self.linears_prediction[i](readout)
            readout_all_layer += self.dropout(tmp)

        # return readout_all_layer
        return torch.sigmoid(readout_all_layer)

class MatData(torch.utils.data.Dataset):
    def __init__(self,num,graph_dir,transform=None):
        self.num = num
        self.transform = transform
        self.path_dir = graph_dir

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        graph = torch.load(os.path.join(self.path_dir,'graph{}.dat'.format(idx)))
        
        if self.transform:
            graph = self.transform(graph)

        return graph
    
def train(num_epochs,trainloader,device,model,optimizer,criterion,model_name = None):
    print('begin to train')
    model.train()
    i = 0
    train_loss_list = []
    train_acc_list = []
    for epoch in range(num_epochs):
        ## training step
        for graphs in trainloader:
            graphs = graphs.to(device)

            ## forward + backprop + loss
            out = model(graphs)
            loss = criterion(out, graphs.y.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()

            ## update model params
            optimizer.step()

            train_running_loss = loss.item()
        
            print('Epoch: {:3} | Batch: {:3}| Loss: {:6.4f}'.format(epoch,i,train_running_loss))

            i = i + 1
            train_loss_list.append(train_running_loss)

    if model_name is not None:
        torch.save(model,model_name)

    return  train_loss_list, i

def test(testloader,device,model,criterion):
    model.eval()
    print('================================================')
    print('begin to test')
    total_num = len(testloader.dataset)
    print('test, num = {}'.format(total_num))

    mat_list = []
    theta_list = []
    total_loss = 0
    percentDiff = np.zeros(total_num)
    i = 0

    for graphs in testloader:
        graphs = graphs.to(device)
        with torch.no_grad():
            outputs = model(graphs)
            loss = criterion(outputs, graphs.y.unsqueeze(1))

            percentDiff[i] = abs( (outputs.item() - graphs.y.item())/graphs.y.item() )
            total_loss += loss.item()
            print('mat_id: {} | predict: {} | y: {} | percent: {:.2%} | loss: {}'.format(graphs.mat_id.item(),outputs.item(),graphs.y.item(),percentDiff[i],loss.item()))
            mat_list.append(graphs.mat_id.item())
            theta_list.append(outputs.item())
            i += 1

    print('total loss sum = {}'.format(total_loss))
    mean = np.mean(percentDiff)
    std = np.std(percentDiff)
    diff_max = np.max(percentDiff)
    print('mean = {:.2%}, std = {:.2%}, max = {:.2%}'.format(mean,std,diff_max))

    return mat_list, theta_list

def run(base_path,mat,values):
    batch_size = 1
    exec_path = './solver '
    keys = ['file','str']

    iters = []
    times = []
    for i in range(len(mat)):
        print("=======================================")
        print("begin to run matrix {}".format(mat[i]))
        # jxp_path = os.path.join(base_path, f'/{mat[i]}/jxp.txt')
        jxp_path = base_path + '{}/jxp.txt'.format(mat[i])
        print(jxp_path)

        vals = [jxp_path,values[i]]
        
        a = JXPExecute.RunningNoLog(exec_path,batch_size,keys,vals)
        a.BatchExec()
        a.CollectInfo()

        iters.append(a.average_iter)
        times.append(a.average_time)
        print("theta = {}, iter = {}, time = {}".format(values[i],a.average_iter,a.average_time))

    print("=======================================")
    iters_sum = sum(iters)
    iters_mean = iters_sum/len(iters)
    print("iteration sum = {}, mean = {}".format(iters_sum,iters_mean))

    times_sum = sum(times)
    times_mean = times_sum/len(times)
    print("time sum = {}, mean = {}".format(times_sum,times_mean))

    default_iter_list = []
    opt_list = []
    list_50 = []

    time_opt = []
    time_25 = []
    time_50 = []
    for i in range(len(mat)):
        # config_path = os.path.join(base_path,f'/{mat[i]}/config.json')
        config_path = base_path +'/{}/config.json'.format(mat[i])
        with open(config_path,'r') as f:
            info = json.load(f)
        default_iter_list.append(info['iter_0.25'])
        opt_list.append(info['min_iter'])
        list_50.append(info['iter_0.5'])

        time_25.append(info['time_vec'][24])
        time_50.append(info['time_vec'][49])
        time_opt.append(min(info['time_vec']))

    default_sum = sum(default_iter_list)
    default_mean = default_sum/len(default_iter_list)
    print("threshold = 0.25, iteration sum = {}, mean = {}; time sum = {}, mean = {}".format(default_sum,default_mean,sum(time_25),sum(time_25)/len(time_25)))

    sum_50 = sum(list_50)
    mean_50 = sum_50/len(list_50)
    print("threshold = 0.5, iteration sum = {}, mean = {}; time sum = {}, mean = {}".format(sum_50,mean_50,sum(time_50),sum(time_50)/len(time_50) ))

    opt_sum = sum(opt_list)
    opt_mean = opt_sum/len(opt_list)
    print("opt, iteration sum = {}, mean = {}; time sum = {}, mean = {}".format(opt_sum,opt_mean,sum(time_opt),sum(time_opt)/len(time_opt) ))

def main():
    num_epochs = 50
    epoch_step = 5
    learning_rate = 0.001
    random_seed = 1
    setup_seed(random_seed)

    dataset = MatData(100,'./GraphData/')

    train_idx = list(range(80))
    test_idx = list(range(80,100))
    
    train_set = torch.utils.data.Subset(dataset,train_idx)
    test_set = torch.utils.data.Subset(dataset,test_idx)

    trainloader = pygdat.DataLoader(train_set,batch_size=8,shuffle=True)
    testloader = pygdat.DataLoader(test_set,batch_size=1,shuffle=False)

    print(torch.cuda.is_available())
    device = torch.device("cuda:0")
    # device = torch.device("cpu")
    model = GNN(4,1,32,1)
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_list, total_batch_num = train(num_epochs,trainloader,device,model,optimizer,criterion)
    # plot_loss(loss_list,total_batch_num,num_epochs,epoch_step,pic_name)

    mat_list, theta_list = test(testloader,device,model,criterion)
    base_path = './MatData/'
    run(base_path,mat_list,theta_list)

if __name__ == '__main__':
    main()
