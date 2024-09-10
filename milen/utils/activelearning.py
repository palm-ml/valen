import torch
import copy

class PLLActiveInteraction:
    def __init__(self, start_epoch=30, end_epoch=500, step_epoch=50, interact_times=10, select_num=100, select_metric="entropy") -> None:
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.step_epoch = step_epoch
        self.interact_idx = 0 
        self.final_interact_times = interact_times
        self.select_num = int(select_num)
        self.select_metric = select_metric
        self.idx_pool = []
        if self.select_metric == "entropy":
            def entropy(x, y, eps=1e-12):
                x = x + 1e-12
                x = x / x.sum(dim=1, keepdim=True)
                return - ((x * torch.log(x)) * y).sum(dim=1)
            self.select_fn = entropy

        if self.select_metric == "margin":
            def margin(x, y):
                x = x*y
                max_x, max_x_idx = x.max(dim=1)
                _x = copy.deepcopy(x)
                _x[torch.arange(0, len(max_x_idx)), max_x_idx] = -1
                submax_x, _ = _x.max(dim=1)
                return submax_x - max_x
            self.select_fn = margin

        if self.select_metric == "maximum":
            def maximum(x, y):
                x = x * y
                return 1 - x.max(dim=1)[0]
            self.select_fn = maximum

    def interact(self, epoch, d, Y):
        if self.interact_idx <= self.final_interact_times:
            if epoch >= self.start_epoch and epoch <= self.end_epoch and (epoch - self.start_epoch) % self.step_epoch == 0:
                self.interact_idx += 1
                torch.set_printoptions(profile='full')
                print("interact: {}".format(self.interact_idx))
                new_d = copy.deepcopy(d)

                metric_array = self.select_fn(d, Y)

                metric_value, metric_idx = metric_array.topk(self.select_num * self.final_interact_times, largest=True)

                new_d[metric_idx, :] = copy.deepcopy(Y[metric_idx, :])
                return new_d
        return d
    
    def interact_v2(self, epoch, d, o, Y, T):
        if self.interact_idx <= self.final_interact_times:
            if epoch >= self.start_epoch and epoch <= self.end_epoch and (epoch - self.start_epoch) % self.step_epoch == 0:
                self.interact_idx += 1
                torch.set_printoptions(profile='full')
                print("interact: {}".format(self.interact_idx))
                new_d = copy.deepcopy(d)
                new_o = copy.deepcopy(o)
                new_Y = copy.deepcopy(Y)
                metric_array = self.select_fn(d, Y)
                metric_value, metric_idx = metric_array.topk(self.select_num* self.final_interact_times, largest=True)
                select_idx = []
                select_num = 0
                num_pool = len(self.idx_pool)
                for idx in metric_idx:
                    if idx in self.idx_pool:
                        continue
                    else:
                        select_idx.append(idx)
                        select_num += 1
                    if select_num >= self.select_num:
                        break
                self.idx_pool += select_idx
                print("change {} samples".format(len(self.idx_pool) - num_pool))
                new_d[select_idx, :] = copy.deepcopy(T[select_idx, :])
                new_o[select_idx, :] = copy.deepcopy(T[select_idx, :])
                new_Y[select_idx, :] = copy.deepcopy(T[select_idx, :])
                return new_Y.cpu().numpy(), new_d, new_o
        return Y.cpu().numpy(), d, o