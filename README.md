# AutoAMG

- `MatGen.py`: Generate matrix data based on [FEALPy](https://github.com/weihuayi/fealpy), then transform the matrix data into graph data for training and testing, and change the format of the matrix for solving through `JXPAMG`
- `GNN.py`: graph neural network architecture, training and testing programs
- `JXPExecute.py`:  call solving  program and deal with the output 
- `JXPGridSearch.py`: find the optimal $\theta$ through grid search 