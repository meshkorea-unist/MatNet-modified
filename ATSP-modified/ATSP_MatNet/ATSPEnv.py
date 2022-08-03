
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

from dataclasses import dataclass
import torch
import copy

from CATSProblemDef import load_predefined_problems, augment_xy_data_by_8_fold


@dataclass
class Reset_State:
    depot_node_xy: torch.Tensor = None
    # shape: (batch, node+1, 2)
    duration_matrix: torch.Tensor = None
    # shape: (batch, node+1, node+1)


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor = None
    POMO_IDX: torch.Tensor = None
    # shape: (batch, pomo)
    selected_count: int = None
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, node_cnt+1)
    finished: torch.Tensor = None
    # shape: (batch, pomo)
    dummy_mask = None
    # shape: (batch, pomo, node_cnt+1)



class ATSPEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.node_cnt = env_params['node_cnt']
        self.pomo_size = env_params['pomo_size']
        self.file_path = env_params['file_path']

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)
        self.depot_node_xy = None
        # shape: (batch, node_cnt+1, 2)
        self.duration_matrix = None
        # shape: (batch, node+1, node+1)
        
        # Dynamic
        ####################################
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~)
        self.ninf_mask = None
        # shape: (batch, pomo, node_cnt+1)
        self.finished = None
        # shape: (batch, pomo)
        self.dummy_mask = None
        # shape: (batch, pomo, node_cnt+1)

        # states to return
        ####################################
        self.reset_state = Reset_State()
        self.step_state = Step_State()

    def load_problems(self, batch_size, aug_factor=1):
        self.batch_size = batch_size
        depot_xy, node_xy, duration_matrix, dummy_mask = load_predefined_problems(batch_size, self.node_cnt, self.file_path)
        depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                depot_node_xy = augment_xy_data_by_8_fold(depot_node_xy)
                duration_matrix = duration_matrix.repeat(8, 1, 1)
                dummy_mask = dummy_mask.repeat(8, 1, 1)
            else:
                raise NotImplementedError

        self.depot_node_xy = depot_node_xy
        # shape: (batch, node_cnt+1, 2)
        self.dummy_mask = dummy_mask
        # shape: (batch, pomo, node_cnt+1)

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

        self.reset_state.depot_node_xy = depot_node_xy
        self.reset_state.duration_matrix = duration_matrix
        
        self.duration_matrix = duration_matrix

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX
        
    def reset(self):
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~)
        self.ninf_mask = copy.deepcopy(self.dummy_mask)
        # shape: (batch, pomo, node_cnt+1)
        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)

        reward = None
        done = False
        return self.reset_state, reward, done

    def pre_step(self):
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        self.step_state.dummy_mask = self.dummy_mask

        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        # selected.shape: (batch, pomo)

        # Dynamic
        ####################################
        self.current_node = selected
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~)

        self.ninf_mask[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')
        # shape: (batch, pomo, node_cnt+1)

        round_error_epsilon = 0.00001

        newly_finished = (self.ninf_mask == float('-inf')).all(dim=2)
        # shape: (batch, pomo)
        self.finished = self.finished + newly_finished
        # shape: (batch, pomo)

        # do not mask depot for finished episode.
        self.ninf_mask[:, :, 0][self.finished] = 0

        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        # returning values
        done = self.finished.all()
        if done:
            reward = -self._get_total_duration()  # note the minus sign!
        else:
            reward = None

        return self.step_state, reward, done

    def _get_total_duration(self):

        node_from = self.selected_node_list
        # shape: (batch, pomo, node)
        node_to = self.selected_node_list.roll(dims=2, shifts=-1)
        # shape: (batch, pomo, node)
        batch_index = self.BATCH_IDX[:, :, None].expand(self.batch_size, self.pomo_size, self.node_cnt)
        # shape: (batch, pomo, node)

        selected_cost = self.duration_matrix[batch_index, node_from, node_to]
        # shape: (batch, pomo, node)
        total_duration = selected_cost.sum(2)
        # shape: (batch, pomo)

        return total_duration
