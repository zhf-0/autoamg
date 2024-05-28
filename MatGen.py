import os
import argparse 
import numpy as np
# from scipy.sparse.linalg import spsolve, inv, norm
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator
import scipy.sparse as sparse
import pandas as pd 
import json
import JXPExecute

from fealpy.decorator import cartesian, barycentric
from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.functionspace import  ParametricLagrangeFiniteElementSpace
from fealpy.boundarycondition import DirichletBC
from fealpy.tools.show import showmultirate, show_error_table

import torch
import torch_geometric.data as pygdat
from torch_geometric.utils import degree

def COO2CSR0(row,col,nnz,coo_i,coo_j,coo_val):
    '''
    coo to csr 
    '''
    csr_i = np.zeros(row+1,dtype=int)
    csr_j = np.zeros(nnz,dtype=int)
    csr_val = np.zeros(nnz,dtype=np.float64)

    for i in range(nnz):
        csr_i[coo_i[i] + 1] += 1

    num_per_row = csr_i.copy()

    for i in range(2,row + 1):
        csr_i[i] = csr_i[i] + csr_i[i-1]

    for i in range(nnz):
        row_idx = coo_i[i]
        begin_idx = csr_i[row_idx]
        end_idx = csr_i[row_idx+1]
        offset = end_idx - begin_idx - num_per_row[row_idx+1]
        num_per_row[row_idx + 1] -= 1

        csr_j[begin_idx + offset] = coo_j[i]
        csr_val[begin_idx + offset] = coo_val[i]

    return csr_i, csr_j, csr_val

class PDE:
    def __init__(self,x0,x1,y0,y1,blockx,blocky):
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        self.xstep = (x1-x0)/blockx 
        self.ystep = (y1-y0)/blocky
        self.coef1 = 10**np.random.uniform(0.0,5.0,(blocky+1,blockx+1))
        self.coef2 = 10**np.random.uniform(0.0,5.0,(blocky+1,blockx+1))

    def domain(self):
        return np.array([self.x0, self.x1,self.y0, self.y1])
    
    @cartesian
    def solution(self, p):
        """ 
		The exact solution 
        Parameters
        ---------
        p : 
        Examples
        -------
        p = np.array([0, 1], dtype=np.float64)
        p = np.array([[0, 1], [0.5, 0.5]], dtype=np.float64)
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.cos(pi*x)*np.cos(pi*y)
        return val # val.shape == x.shape

    @cartesian
    def source(self, p):
        """ 
		The right hand side of convection-diffusion-reaction equation
        INPUT:
            p: array object,  
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = 12*pi*pi*np.cos(pi*x)*np.cos(pi*y) 
        val += 2*pi*pi*np.sin(pi*x)*np.sin(pi*y) 
        val += np.cos(pi*x)*np.cos(pi*y)*(x**2 + y**2 + 1) 
        val -= pi*np.cos(pi*x)*np.sin(pi*y) 
        val -= pi*np.cos(pi*y)*np.sin(pi*x)
        return val

    @cartesian
    def gradient(self, p):
        """ 
		The gradient of the exact solution 
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = -pi*np.sin(pi*x)*np.cos(pi*y)
        val[..., 1] = -pi*np.cos(pi*x)*np.sin(pi*y)
        return val # val.shape == p.shape

    @cartesian
    def diffusion_coefficient(self, p):
        x = p[..., 0]
        y = p[..., 1]
        xidx = x//self.xstep
        xidx = xidx.astype(np.int)
        yidx = y//self.ystep 
        yidx = yidx.astype(np.int)

        shape = p.shape+(2,)
        val = np.zeros(shape,dtype=np.float64)
        val[...,0,0] = self.coef1[xidx,yidx]
        # val[...,0,0] = 10.0
        val[...,0,1] = 1.0
        val[...,1,0] = 1.0
        val[...,1,1] = self.coef2[xidx,yidx]
        # val[...,1,1] = 2.0
        return val

    @cartesian
    def convection_coefficient(self, p):
        return np.array([-1.0, -1.0], dtype=np.float64)

    @cartesian
    def reaction_coefficient(self, p):
        x = p[..., 0]
        y = p[..., 1]
        return 1 + x**2 + y**2

    @cartesian
    def dirichlet(self, p):
        return self.solution(p)



class CreateAllData:
    def __init__(self,num):
        self.num = num
        self.root_path = './MatData/'
        self.full_path = os.path.join(self.root_path,'full')
        self.succ_path = os.path.join(self.root_path,'succ')
        self.fig_path = os.path.join(self.succ_path,'pic')
        os.makedirs(self.full_path,exist_ok=True)
        os.makedirs(self.succ_path,exist_ok=True)
        os.makedirs(self.fig_path,exist_ok=True)

        np.random.seed(0)
        self.mesh = np.random.randint(512,1024,num)
        self.block = np.random.randint(2,9,num)
        self.nx = 0
        self.ny = 0
        self.blockx = 0
        self.blocky = 0

        dirs_list = os.listdir(self.succ_path)
        if 'pic' in dirs_list:
            dirs_list.remove('pic')
        self.num_succ = len(dirs_list)

    def Process(self):
        print('begin to process')
        for i in range(self.num):
            np.random.seed(i)
            self.nx = int(self.mesh[i])
            self.ny = self.nx
            self.blockx = int(self.block[i])
            self.blocky = self.blockx

            info = {}
            info['seed'] = i
            info['meshx'] = self.nx
            info['meshy'] = self.ny
            info['blockx'] = self.blockx
            info['blocky'] = self.blocky

            json_path = os.path.join(self.full_path,'config{}.json'.format(i))
            if os.path.exists(json_path):
                continue

            print('========================================================')
            print(f'begin to run, seed = {i}')

            print('begin to generate matrix')
            A = self.GenerateMat()

            print('begin to write matrix for jxpamg')
            jxp_path = os.path.join(self.root_path,'mat.jxp')
            self.WriteJxpMat(A,jxp_path)

            print('begin to solve matrix by jxpamg')
            time_vec, iter_vec, resi_vec = self.RunJXP(jxp_path)

            print('begin to analyze statistic info')
            threshold = list(np.linspace(0.01,0.99,99,endpoint=True))
            min_iter = min(iter_vec)
            min_idx = iter_vec.index(min_iter)
            info['min_iter'] = min_iter
            info['min_thres'] = threshold[min_idx]

            max_iter = max(iter_vec)
            max_idx = iter_vec.index(max_iter)
            info['max_iter'] = max_iter
            info['max_thres'] = threshold[max_idx]
            
            info['iter_0.25'] = iter_vec[24]
            info['iter_0.5'] = iter_vec[49]

            tmp_iter = np.array(iter_vec)
            info['iter_mean'] = tmp_iter.mean()
            info['iter_var'] = tmp_iter.var()

            info['time_vec'] = time_vec
            info['iter_vec'] = iter_vec
            info['resi_vec'] = resi_vec

            if tmp_iter.var() > 100:
                print('=================================')
                print('a successful one ')
                info['status'] = True
                item_path = os.path.join(self.succ_path,f'{self.num_succ}')
                self.num_succ += 1
                os.makedirs(item_path,exist_ok=True)

                print('begin to write json  {}'.format(i))
                succ_json_path = os.path.join(item_path,'config.json')
                with open(succ_json_path,'w') as f:
                    json.dump(info,f,indent=4)

                print('begin to write coo   {}'.format(i))
                succ_mat_path = os.path.join(item_path,'coomat.txt')
                self.WriteCooMat(A,succ_mat_path)

                print('begin to write graph {}'.format(i))
                succ_graph_path = os.path.join(item_path,'graph.dat')
                graph = self.CreateGraph(A)
                graph.mat_id = i
                torch.save(graph,succ_graph_path)

                print('begin to plot figure')
                succ_fig_path = os.path.join(self.fig_path, f'{self.num_succ}.png')
                Plot(info,succ_fig_path)
            else:
                info['status'] = False

            with open(json_path,'w') as f:
                json.dump(info,f,indent=4)

            print('delete the jxpamg matrix')
            os.remove(jxp_path)

    def RunJXP(self,mat_path):
        batch_size = 1
        exec_path = './solver '
        keys = ['file','str']

        JXPtime = []
        JXPiteration = []
        JXPresidual = []

        threshold = list(np.linspace(0.01,0.99,99,endpoint=True))
        for thres in threshold:
            print(f'  - begin to run when theta = {thres}')
            vals = [mat_path, thres]

            a = JXPExecute.RunningNoLog(exec_path,batch_size,keys,vals)
            a.BatchExec()
            a.CollectInfo()

            JXPtime.append(a.average_time)
            JXPiteration.append(a.average_iter)
            JXPresidual.append(a.average_resi)

        return JXPtime, JXPiteration, JXPresidual

    def GenerateMat(self):
        pde = PDE(0,1,0,1,self.blockx,self.blocky)
        domain = pde.domain()
        mesh = MF.boxmesh2d(domain, nx=self.nx, ny=self.ny, meshtype='quad',p=1)

        # space = LagrangeFiniteElementSpace(mesh, p=1)
        space = ParametricLagrangeFiniteElementSpace(mesh, p=1)
        NDof = space.number_of_global_dofs()
        uh = space.function() 	
        A = space.stiff_matrix(c=pde.diffusion_coefficient)
        # B = space.convection_matrix(c=pde.convection_coefficient)
        # M = space.mass_matrix(c=pde.reaction_coefficient)
        F = space.source_vector(pde.source)
        # A += B 
        # A += M
        
        bc = DirichletBC(space, pde.dirichlet)
        A, F = bc.apply(A, F, uh)

        return A

    def WriteJxpMat(self,coomat,mat_path):
        row = coomat.shape[0]
        nnz = coomat.nnz
        coo_i, coo_j = coomat.nonzero()
        coo_val = coomat.data

        # coo to csr 
        print('coo to csr')
        csr_i,csr_j,csr_val = COO2CSR0(row,row,nnz,coo_i,coo_j,coo_val)

        # csr to jxpamg file 
        print('csr to jxpamg file')
        output = [None]*(2+row+2*nnz)
        template = "{} \n"
        output[0] = template.format(row)
        idx = 1
        for i in range(row+1):
            output[idx] = template.format(csr_i[i])
            idx += 1

        for i in range(nnz):
            output[idx] = template.format(csr_j[i])
            idx += 1

        for i in range(nnz):
            output[idx] = template.format(csr_val[i])
            idx += 1

        with open(mat_path,'w') as f:
            f.writelines(output)


    def WriteCooMat(self,scicoo,mat_path):
        row = scicoo.shape[0]
        nnz = scicoo.nnz
        coo_i, coo_j = scicoo.nonzero()
        coo_val = scicoo.data

        output = [None]*(nnz + 1)
        template = "{} {} {} \n"
        output[0] = template.format(row,row,nnz)
        for i in range(nnz):
            idx = i + 1
            output[idx] = template.format(coo_i[i],coo_j[i],coo_val[i])

        with open(mat_path,'w') as f:
            f.writelines(output)

    def CreateGraph(self,A):
        row, col = A.nonzero()
        edge_weight = torch.zeros(A.nnz,dtype=torch.float32)
        for i in range(A.nnz):
            edge_weight[i] = abs(A[row[i],col[i]])

        row = torch.from_numpy(row.astype(np.int64))
        col = torch.from_numpy(col.astype(np.int64))
        x = degree(col,A.shape[0],dtype=torch.float32).unsqueeze(1)
        edge_index = torch.stack((row,col),0)
        # y = torch.tensor(y,dtype=torch.float32)

        # graph = pygdat.Data(x=x,edge_index = edge_index,edge_weight = edge_weight,y = y)
        graph = pygdat.Data(x=x,edge_index = edge_index,edge_weight = edge_weight)

        return graph


def Plot(info,fig_path):
    x = np.linspace(0.01,0.99,99,endpoint=True)
    y = np.array(info['iter_vec'])
    opt_x = info['min_thres']
    opt_iter = info['min_iter']

    plt.figure()
    plt.xlabel(r"$\theta$",fontsize=20)
    plt.ylabel("iterations",fontsize=20)
    
    x_major_locator=MultipleLocator(0.1)
    y_major_locator=MultipleLocator(50)
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)

    plt.bar(x,y,width=0.005, color='blue')
    plt.bar(opt_x,opt_iter,width=0.005,color='red')

    plt.savefig(fig_path)
    # plt.show()
    plt.close()

def main():
    mat = CreateAllData(100)
    mat.Process()


if __name__ == '__main__':
    main()
