#!/usr/bin/env python3
import itertools
import operator
import numpy as np
import pandas as pd
import JXPExecute

def FinalProcess(keys,vals,grid_path):
    batch_size = 1
    exec_path = './solver'

    grid_contents = {}
    for key in keys:
        grid_contents[key] = []

    total_par  = []
    total_time = []
    total_iter = []
    total_resi = []
    for item in itertools.product(*vals):
        # a = JXPExecute.Running(exec_path,batch_size,keys,item)
        a = JXPExecute.RunningNoLog(exec_path,batch_size,keys,item)
        a.BatchExec()
        a.CollectInfo()
        # out = "when parameter = {}; num of iter = {}; the elapsed time = {}; the residual = {}; the log path is : {}".format(item,a.average_iter,a.average_time,a.average_resi,a.log_path)
        out = "when parameter = {}; num of iter = {}; the elapsed time = {}; the residual = {}; ".format(item,a.average_iter,a.average_time,a.average_resi)
        print(out,flush=True)

        total_par.append(item)
        total_time.append(a.average_time)
        total_iter.append(a.average_iter)
        total_resi.append(a.average_resi)
        for i in range(len(keys)):
            grid_contents[keys[i]].append(item[i]) 


    min_index, min_number = min(enumerate(total_iter), key=operator.itemgetter(1))

    out1 = "\n===================================================="
    opt_out = "when parameter = {}; num of iter = {}; the elapsed time = {}; the residual = {}".format(total_par[min_index],total_iter[min_index],total_time[min_index],total_resi[min_index])
    out2 = "====================================================\n"

    total_iter.append(total_iter[min_index])
    total_time.append(total_time[min_index])
    total_resi.append(total_resi[min_index])

    for k in range(len(keys)):
        grid_contents[keys[k]].append(total_par[min_index][k])

    grid_contents['iter'] = total_iter
    grid_contents['time'] = total_time
    grid_contents['residual'] = total_resi

    if grid_path != 0:
        res = pd.DataFrame(grid_contents)
        res.to_excel(grid_path,index=False)
    

    print(out1)
    print(opt_out)
    print(out2)

if __name__ == "__main__":


    # this is for threshold with different dir
    #=============================================================
    keys = ['file','str']
    
    # dir_list = [0,1,2]
    dir_list = [i for i in range(90,100)]

    list2 = list(np.linspace(0.01,0.99,99,endpoint=True))

    for item in dir_list:
        file_path = './data/{}/'.format(item)
        vals = [[file_path],list2]
        grid_file = './data/{}/results.xlsx'.format(item)
        FinalProcess(keys,vals,grid_file)
    #=============================================================

