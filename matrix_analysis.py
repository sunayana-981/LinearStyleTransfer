#read the files in Matrices\ and convert it to a numpy array
import torch
import numpy as np
import os
import sys  

def read_file(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
    return lines

def write_file(file_name, lines):
    with open(file_name, 'w') as f:
        f.writelines(lines)

def read_matrix(file_name):

    lines = read_file(file_name)
    matrix = []
    for line in lines:
        matrix.append([float(x) for x in line.split()])
    return np.array(matrix)

def write_matrix(file_name, matrix):

    lines = []
    for row in matrix:
        lines.append(' '.join([str(x) for x in row]) + '\n')
    write_file(file_name, lines)

def read_matrices(folder_name):

    matrices = []
    for file_name in os.listdir(folder_name):
        matrices.append(read_matrix(os.path.join(folder_name, file_name)))
    return matrices

def write_matrices(folder_name, matrices):

    for i, matrix in enumerate(matrices):
        write_matrix(os.path.join(folder_name, 'matrix_' + str(i) + '.txt'), matrix)

def read_tensor(file_name):

    matrix = read_matrix(file_name)
    return torch.tensor(matrix)

def write_tensor(file_name, tensor):

    matrix = tensor.numpy()
    write_matrix(file_name, matrix)

def read_tensors(folder_name):

    tensors = []
    for file_name in os.listdir(folder_name):
        tensors.append(read_tensor(os.path.join(folder_name, file_name)))
    return tensors

def write_tensors(folder_name, tensors):

    for i, tensor in enumerate(tensors):
        write_tensor(os.path.join(folder_name, 'tensor_' + str(i) + '.txt'), tensor)

def main():

    input_folder="Matrices/"
    output_folder="Matrices1/"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    matrices = read_matrices(input_folder)
    write_matrices(output_folder, matrices)

if __name__ == '__main__':

    main()

    



