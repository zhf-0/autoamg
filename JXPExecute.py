#!/usr/bin/env python3
import re
import os 
import sys
import subprocess
import time
import random

def IsNumerical(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

class Running:
    def __init__(self,e_path,size,keys,vals):
        log_name = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
        log_path = os.path.join(os.getcwd(),'log',log_name)
        count = 1
        while os.path.exists(log_path):
            log_path = log_path+'_'+str(count) 
            count = count + 1
        os.makedirs(log_path)
        
        self.log_path = log_path
        self.batch_size = size 
        self.exec_path = e_path 

        self.config = {}
        self.keys = keys
        self.vals = vals
        self.output_config = os.path.join(self.log_path,'config.dat')

        self.max_it = 1000
        self.max_time = 10000.0
        self.iterations = []
        self.elapse_times = []
        self.residuals = []
        self.average_time = 0
        self.average_iter = 0
        self.average_resi = 0

    def OutputConfig(self):
        for i in range(len(self.keys)):
            self.config[self.keys[i]] = self.vals[i]

        contents = []
        for k,v in self.config.items():
            line = '{:<25} = {:<25} \n'.format(k,v)
            contents.append(line)

        with open(self.output_config,'w') as f:
            f.writelines(contents)

    def BatchExec(self):
        self.OutputConfig()
        cmd = self.exec_path 
        template = " -{} {} "
        for i in range(len(self.keys)):
            cmd += template.format(self.keys[i],self.vals[i])

        # print(cmd)

        for i in range(self.batch_size):
            try:
                run_output = subprocess.check_output(cmd,shell=True)
            except subprocess.CalledProcessError as error:
                print('\033[31m running fail! :','check the log dir {} \033[0m'.format(self.log_path))
                print(error.output.decode('utf-8'))
                # contents = error.output.decode('utf-8')
                contents = ['Number of iterations = {} with relative residual 0.\n'.format(self.max_it),'AMG_Krylov method totally costs {} seconds\n'.format(self.max_time)]
            else:
                contents = run_output.decode('utf-8')
            finally:
                local_log_path = os.path.join(self.log_path,str(i)+'.log')
                with open(local_log_path,'w') as f:
                    f.write(contents)
                    change_list = ['\n','='*75,'\n']
                    for k in range(len(self.keys)):
                        tmp_line = '{:<25} = {:<25} \n'.format(self.keys[k],self.vals[k])
                        change_list.append(tmp_line)

                    f.writelines(change_list)

                lines = contents.split('\n')
                for i, line in enumerate(lines):
                    if re.search('num_iterations',line):
                        self.iterations.append(int(line.split()[-1]))
                    if re.search('final_res_norm',line):
                        self.residuals.append(float(line.split()[-1]))
                    if re.search('Total',line):
                        self.elapse_times.append(float(line.split(',')[-2]))

    def CollectInfo(self):
        # the target is iteration, if the target is elapsed time, change the if-else condition as follow
        result_len = len(self.iterations)
        if result_len == 0:
            print('\033[31m can not collect the info : the len of result is 0, check the log dir {} \033[0m'.format(self.log_path))
            self.average_iter = self.max_it
            self.average_time = self.max_time
            self.average_resi = 0.0
        else:
            for i in range(result_len):
                if not IsNumerical(self.iterations[i]):
                    print('\033[31m the value is not a number, check the log dir {} \033[0m'.format(self.log_path))
                    self.iterations[i] = self.max_it
                elif self.iterations[i] == 0: 
                    print('\033[31m the value is 0, check the log dir {} \033[0m'.format(self.log_path))
                elif abs( self.iterations[i] - self.iterations[0] ) > 5:
                    print("be careful! the iterations[{}] changes rapidly".format(i))

            self.average_time = sum(self.elapse_times) / result_len
            self.average_iter = sum(self.iterations) / result_len
            self.average_resi = sum(self.residuals) / result_len



class RunningNoLog:
    def __init__(self,e_path,size,keys,vals):
        
        self.batch_size = size 
        self.exec_path = e_path 

        self.config = {}
        self.keys = keys
        self.vals = vals

        self.max_it = 1000
        self.max_time = 10000.0
        self.iterations = []
        self.elapse_times = []
        self.residuals = []
        self.average_time = 0
        self.average_iter = 0
        self.average_resi = 0

    def BatchExec(self):
        cmd = self.exec_path 
        template = " -{} {} "
        for i in range(len(self.keys)):
            cmd += template.format(self.keys[i],self.vals[i])

        # print(cmd)

        for i in range(self.batch_size):
            try:
                run_output = subprocess.check_output(cmd,shell=True)
            except subprocess.CalledProcessError as error:
                print('\033[31m running fail! :','check the parameters {} \033[0m'.format(self.vals))
                print(error.output.decode('utf-8'))
                sys.exit(1)
            else:
                contents = run_output.decode('utf-8')
                lines = contents.split('\n')
                for i, line in enumerate(lines):
                    if re.search('num_iterations',line):
                        self.iterations.append(int(line.split()[-1]))
                    if re.search('final_res_norm',line):
                        self.residuals.append(float(line.split()[-1]))
                    if re.search('Total',line):
                        self.elapse_times.append(float(line.split(',')[-2]))

    def CollectInfo(self):
        # the target is iteration, if the target is elapsed time, change the if-else condition as follow
        result_len = len(self.iterations)
        if result_len == 0:
            print('\033[31m can not collect the info : the len of result is 0, check the parameters {} \033[0m'.format(self.vals))
            self.average_iter = self.max_it
            self.average_time = self.max_time
            self.average_resi = 0.0
        else:
            for i in range(result_len):
                if not IsNumerical(self.iterations[i]):
                    print('\033[31m the value is not a number, check the parameters {} \033[0m'.format(self.vals))
                    self.iterations[i] = self.max_it
                elif self.iterations[i] == 0: 
                    print('\033[31m the value is 0, check the parameters {} \033[0m'.format(self.vals))
                elif abs( self.iterations[i] - self.iterations[0] ) > 5:
                    print("be careful! the iterations[{}] changes rapidly".format(i))

            self.average_time = sum(self.elapse_times) / result_len
            self.average_iter = sum(self.iterations) / result_len
            self.average_resi = sum(self.residuals) / result_len
