
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
import json
import os
import numpy as np


def load_predefined_problems(batch_size, node_cnt, file_path):
    min_coords = torch.Tensor([919557.27446633, 1883094.34362266])
    max_coords = torch.Tensor([ 986881.66391917, 1976116.0223603 ])
    ################################
    # "json" type
    ################################
    depot_xy = torch.zeros((batch_size, 1, 2))
    # shape: (batch, 1, 2)
    node_xy = torch.zeros((batch_size, node_cnt, 2))
    # shape: (batch, node_cnt, 2)
    duration_matrix = torch.full((batch_size, node_cnt+1, node_cnt+1), 0)
    # shape: (batch, node_cnt+1, node_cnt+1)
    dummy_mask = torch.zeros((batch_size, node_cnt+1, node_cnt+1))
    # shape: (batch, node_cnt+1, node_cnt+1)

    cnt = 0
    for name in np.random.permutation(list(os.listdir(file_path))):
        try:
            with open(os.path.join(file_path, name)) as f:
                data = json.load(f)
        except Exception as e:
            print(e)
            continue
        raw_node_xy = data['node_xy']
        if len(raw_node_xy) > node_cnt:
            continue

        depot_xy[cnt, :, :] = (max_coords - torch.Tensor(data['depot_xy']))/(max_coords-min_coords)

        node_xy[cnt, :len(raw_node_xy)] = (max_coords - torch.Tensor(raw_node_xy))/(max_coords-min_coords)

        
        dummy_mask[cnt, len(raw_node_xy)+1:, :] = float('-inf')
        dummy_mask[cnt, :, len(raw_node_xy)+1:] = float('-inf')

        duration_matrix[cnt, :len(raw_node_xy)+1, :len(raw_node_xy)+1] = torch.Tensor(data['durations'])
        #print(duration_matrix[duration_matrix>1e4])
        #if len(duration_matrix[duration_matrix>1e4]) != 0:
        #    print(name)
        #    raise

        cnt += 1
        if cnt == batch_size:
            break

    return depot_xy, node_xy, duration_matrix, dummy_mask

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

    return 
