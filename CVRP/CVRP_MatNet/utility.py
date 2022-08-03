import folium
import numpy as np
import torch
from pyproj import Transformer

def transform(xx, yy, frm=5179, to=4326):
    transformer = Transformer.from_crs("EPSG:{}".format(frm), "EPSG:{}".format(to))
    tmp = transformer.transform(xx, yy)
    return np.stack(tmp, axis=1)

def show_polyline_folium(coords, depot=None, utm=False, save = False, out_name='resuts'):
    if not isinstance(coords, np.ndarray):
        coords = np.array(coords)
    if utm:
        coords = transform(coords[:,1], coords[:,0])
    if depot is None:
        depot = coords.mean(axis=0)
    m = folium.Map(location=depot, zoom_start=13)
    folium.PolyLine(locations = coords, tooltip='Polyline').add_to(m)
    if save:
        m.save(outfile= "test"+'.html')
    else:
        return m
    
    
    
class InferenceManager:
    def __init__(self, solution):
        self.problem = solution['problem']
        self.selected_node_list = solution['selected_node']
        self.duration_matrix = solution['duration_matrix']
        
        self.batch_size = solution['problem'].shape[0]
        self.augmented_batch_size = solution['selected_node'].shape[0]
        self.augment_size = self.augmented_batch_size // self.batch_size
        self.pomo_size = solution['selected_node'].shape[1]
        
        self.BATCH_IDX = torch.arange(self.batch_size).repeat(self.augment_size)[:, None].expand(self.augmented_batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)
        
        self._precompute()        
        
    def show_best(self, instance_index):
        aug_index = self.aug_indices[instance_index]
        pomo_index = self.pomo_indices[aug_index, instance_index]
        tmp = np.array(self.problem[instance_index][self.selected_node_list[aug_index*20 + instance_index]])
        tmp=tmp[pomo_index]
        result=[i for i in tmp if (i[0]!=tmp[0][0]) or (i[1]!=tmp[0][1])]
        print(len(result))
        print(self.max_aug_pomo_reward[instance_index])
        return result
    
    
    def _precompute(self):
        node_from = self.selected_node_list
        # shape: (batch, pomo, selected_list_length)
        node_to = self.selected_node_list.roll(dims=2, shifts=-1)
        # shape: (batch, pomo, selected_list_length)
        batch_index = self.BATCH_IDX[:, :, None].expand(self.augmented_batch_size, self.pomo_size, node_to.shape[2])
        # shape: (batch, pomo, selected_list_length)

        selected_cost = self.duration_matrix[batch_index, node_from, node_to]
        # shape: (batch, pomo, node)
        self.total_distance = selected_cost.sum(2)
        # shape: (batch, pomo)
    
        aug_reward = torch.where(self.total_distance==0,100000,self.total_distance).reshape(self.augment_size, self.batch_size, self.pomo_size)
        # shape: (augmentation, batch, pomo)

        self.max_pomo_reward, self.pomo_indices = aug_reward.min(dim=2)  # get best results from pomo
        # shape: (augmentation, batch)
        #print(pomo_indices)
        #print(pomo_indices.shape)
        #print(aug_reward)
        #print(aug_reward[np.tile(np.arange(8),20),np.tile(np.arange(20),8),pomo_indices.reshape(-1)].reshape(8,20))
        self.max_aug_pomo_reward, self.aug_indices = self.max_pomo_reward.min(dim=0)  # get best results from augmentation
        #print(max_pomo_reward[aug_indices,torch.arange(20)])
        #print(max_aug_pomo_reward)
    