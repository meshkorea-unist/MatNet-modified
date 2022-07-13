
"""
The MIT License

Copyright (c) 2021 MatNet

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.



THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import torch
import orjson
import os


def load_predefined_problems(batch_size, node_cnt, file_path):

    ################################
    # "json" type
    ################################
    depot_xy = torch.zeros((batch_size, 1, 2))
    # shape: (batch, 1, 2)
    node_xy = torch.zeros((batch_size, node_cnt, 2))
    # shape: (batch, node_cnt, 2)
    duration_matrix = torch.full((batch_size, node_cnt+1, node_cnt+1), float('inf'))
    # shape: (batch, node_cnt+1, node_cnt+1)
    dummy_mask = torch.zeros((batch_size, node_cnt, node_cnt+1))
    # shape: (batch, node_cnt, node_cnt+1)

    cnt = 0
    real_node_sizes = []
    for file in os.walk(file_path):
        # TODO
        with open(file) as f:
            data = orjson.load(f)
        raw_node_xy = data['node_xy']
        if len(raw_node_xy) > node_cnt:
            continue
        
        depot_xy[cnt, :, :] = torch.Tensor(data['depot_xy'])

        node_xy[cnt, :len(raw_node_xy)] = raw_node_xy

        dummy_mask[cnt, len(raw_node_xy):, len(raw_node_xy)+1:] = float('inf')

        duration_matrix[cnt, :len(raw_node_xy)+1, :len(raw_node_xy)+1] = torch.Tensor(data['duration_matrix'])

        cnt += 1
        if cnt == batch_size:
            break

    '''
    if problem_size == 20:
        demand_scaler = 30
    elif problem_size == 50:
        demand_scaler = 40
    elif problem_size == 100:
        demand_scaler = 50
    else:
        raise NotImplementedError
    '''
    demand_scaler = 5000

    node_demand = torch.randint(1, 10, size=(batch_size, node_cnt)) / float(demand_scaler)
    # shape: (batch, problem)
    
    real_node_sizes = torch.Tensor(real_node_sizes, dtype=torch.int)
    # shape: (batch)

    return depot_xy, node_xy, duration_matrix, node_demand, dummy_mask

def augment_xy_data_by_8_fold(xy_data):
    # xy_data.shape: (batch, N, 2)

    x = xy_data[:, :, [0]]
    y = xy_data[:, :, [1]]
    # x,y shape: (batch, N, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_xy_data = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, N, 2)

    return aug_xy_data