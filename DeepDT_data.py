import numpy as np

import torch

from scipy import sparse as sp

import os
import torch.nn.functional as f
import open3d
from time import *
from scipy.spatial import cKDTree

class ScanData:
    def __init__(self):

        self.pc = None
        self.scan_name = None
        self.data_para = None
        self.cell_vertex_idx = None
        self.adj = None
        self.ref_label = None



    def load_full_scan(self, file_path, cfg):
        print('file_path', file_path)
        self.pc = open3d.read_point_cloud(os.path.join(file_path, "sampled_points.ply"))

        file_names = dict()
        file_names['output_tetrahedron_adj'] = (os.path.join(file_path, "output_tetrahedron_adj"))
        file_names['output_cell_vertex_idx'] = (os.path.join(file_path, "output_cell_vertex_idx.txt"))
        file_names['ref_label'] = (os.path.join(file_path, "ref_point_label.txt"))

        l = dict()
        l['ref_label'] = np.fromfile(file_names['ref_label'], dtype=np.int32, sep=' ').reshape(-1, 7)
        l['adj_mat'] = np.fromfile(file_names['output_tetrahedron_adj'], dtype=np.int32, sep=' ').reshape(-1, 4)
        l['cell_vertex_idx'] = np.fromfile(file_names['output_cell_vertex_idx'], dtype=np.int32, sep=' ').reshape(-1, 4)

        self.cell_vertex_idx = (l['cell_vertex_idx'])
        self.adj = l['adj_mat']
        self.ref_label = l['ref_label'][:, 0:cfg["ref_num"]]
        print("loaded " + file_path.split('/')[-2])


class DeepDT_Data(object):
    def __init__(self):
        self.label_weights = None


        self.cell_vertex_idx = None
        self.adj = None
        self.adj_idx = None
        self.ref_label = None
        self.data_name = ""

        self.neigh_idx = []
        self.sub_idx = []
        self.interp_idx = []
        self.depth = []
        self.batch_feature = []


    def keys(self):
        keys = [key for key in self.__dict__.keys() if self[key] is not None]
        keys = [key for key in keys if key[:2] != '__' and key[-2:] != '__']
        return keys

    def __getitem__(self, key):
        r"""Gets the data of the attribute :obj:`key`."""
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        """Sets the attribute :obj:`key` to :obj:`value`."""
        setattr(self, key, value)

    def to(self, device):
        device_data = DeepDT_Data()
        self_keys = self.keys()
        for key in self_keys:
            if isinstance(self[key], list):
                for e in self[key]:
                    if torch.is_tensor(e):
                        if e.type() == 'torch.IntTensor':
                            device_data[key].append(e.to(device, dtype=torch.long))
                        else:
                            device_data[key].append(e.to(device))
                    else:
                        device_data[key].append(e)
            elif torch.is_tensor(self[key]):
                if self[key].type() == 'torch.IntTensor':
                    device_data[key] = self[key].to(device, dtype=torch.long)
                else:
                    device_data[key] = self[key].to(device)
        return device_data


def create_full_data(scan, cfg):
    out = DeepDT_Data()
    out.data_name = scan.scan_name

    current_xyz = torch.from_numpy(np.asarray(scan.pc.points)).float()  
    current_normals = torch.from_numpy(np.asarray(scan.pc.normals)).float()  
    current_normals = f.normalize(current_normals, p=2, dim=1)

    for i in range(cfg["num_layers"]):
        neigh_idx = get_neighbor_idx(current_xyz.numpy(), current_xyz.numpy(), cfg["k_n"])  
        neigh_normals = current_normals[neigh_idx]  
        neighbor_xyz = current_xyz[neigh_idx]  

        depth = get_depth(current_xyz, neigh_normals, neighbor_xyz, cfg["k_n"])
        depth = depth_normalize(depth)
        if cfg["use_normal"]:
            relative_normal = get_normal(current_normals, neigh_normals)
            batch_feature = torch.cat((depth.unsqueeze(2), relative_normal), dim=2)
        else:
            batch_feature = depth.unsqueeze(2)


        if not cfg["is_training"]:
            torch.manual_seed(0)
        sub_idx = torch.randperm(current_xyz.shape[0] // cfg['sub_sampling_ratio'][i])

        sub_points = current_xyz[sub_idx] 
        sub_normals = current_normals[sub_idx]  
        print(sub_idx.size())
        print(current_xyz[sub_idx].size())

        pool_i = neigh_idx[sub_idx]  
        up_i = get_neighbor_idx(sub_points.numpy(), current_xyz.numpy(), 1)

        out.neigh_idx.append(neigh_idx)
        out.sub_idx.append(pool_i)
        out.interp_idx.append(up_i)
        out.batch_feature.append(batch_feature)
        current_xyz = sub_points
        current_normals = sub_normals

    if cfg["use_label"]:
        tmp_labels = np.asarray(scan.ref_label) - 1
        out.ref_label = torch.from_numpy(tmp_labels)
        if scan.data_para is not None:
            out.label_weights = torch.from_numpy(scan.data_para.class_weights)

    out.cell_vertex_idx = torch.from_numpy(np.asarray(scan.cell_vertex_idx))

    out.adj_idx = torch.from_numpy(np.asarray(scan.adj))

    adj_rows = np.repeat(np.arange(scan.adj.shape[0]), 4)
    adj_cols = np.reshape(scan.adj, adj_rows.size)

    adj = sp.coo_matrix(
        (np.ones(adj_rows.shape[0]), (adj_rows, adj_cols)),
        shape=[scan.adj.shape[0], scan.adj.shape[0]])

    out.adj = normalize(adj + sp.eye(adj.shape[0]))
    return out


def get_neighbor_idx(pc, query_pts, k):


    print("Tree start!")
    t1 = time()
    kdtree = cKDTree(pc)                                
    t = time()
    print("Finish! Time consumption = ", t - t1, "s")
    print("Tree finish!")
    print("Find its K nearest neighbors.")

    (x, idx) = kdtree.query(query_pts, k, n_jobs=-1)   
    t2 = time()
    print(idx.shape) 

    print("Finish! Time consumption = ", t2 - t, "s")
    idxs = torch.from_numpy(idx)

    return idxs       


def get_depth(point_xyz, neighbor_normals, neighbor_xyz, k):


    print(neighbor_xyz.size())

    
                                                                 
    relative_xyz = point_xyz.unsqueeze(1) - neighbor_xyz       

    depth = torch.mul(neighbor_normals, relative_xyz)

    depth = torch.sum(depth, 2)                           

    return depth


def get_normal(point_normals, neighbor_normals):

    print(point_normals.size())
    depth = torch.sum(torch.mul(neighbor_normals, point_normals.unsqueeze(1)), 2, keepdim=True) 
    normals1 = torch.mul(depth, neighbor_normals) 
    normals2 = point_normals.unsqueeze(1) - normals1
    relative_normals = torch.cat((normals1, normals2), dim=2)

    return relative_normals


def depth_normalize(depth):

    max_depth = torch.clamp(torch.max(torch.abs(depth), 1, keepdim=True)[0], min=1e-12)
    depth = depth / max_depth

    return depth


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)







