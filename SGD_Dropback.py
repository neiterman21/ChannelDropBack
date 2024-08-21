import torch
from torch.optim.optimizer import Optimizer, required
import numpy as np
class StepLR_Dropback:
    def __init__(self, optimizer,milestones,dropback_rate_init, dropback_rate_final, dropback_rate_channels):       
        self.optimizer                          = optimizer
        self.dropback_rate_final                = dropback_rate_final  
        self.dropback_rate_init                 = dropback_rate_init 
        self.optimizer.dropback_rate            = self.dropback_rate_init  
        self.optimizer.dropback_rate_channels   = dropback_rate_channels  
        self.milestones                         = milestones
        self.epoch                              = 0 

    def step(self):
        if self.epoch   > self.milestones[1]:            
            self.optimizer.dropback_rate=self.dropback_rate_final
        elif self.epoch > self.milestones[0]:            
            self.optimizer.dropback_rate=self.dropback_rate_final/2 
        else:
            self.optimizer.dropback_rate=self.dropback_rate_init
        self.epoch =self.epoch+1
 

class SGD_Dropback(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + g_{t+1}, \\
                p_{t+1} & = p_{t} - \text{lr} * v_{t+1},
            \end{aligned}

        where :math:`p`, :math:`g`, :math:`v` and :math:`\mu` denote the 
        parameters, gradient, velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + \text{lr} * g_{t+1}, \\
                p_{t+1} & = p_{t} - v_{t+1}.
            \end{aligned}

        The Nesterov version is analogously modified.
    """

    def __init__(self, model, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False,dropback=False,skip_init_layers=4):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        self.dropback_rate    = 0
        self.skip_init_layers = skip_init_layers        
        if dropback==False:
            params=model.parameters()
        else:                        
            list_dropback       = []
            list_no_dropback    = []
            list_names_dropback =[]
            list_names_ndropback =[]
            no_dropback_list    = ["bias", "bn"]
            layer_num           = 0
            for n,p in model.named_parameters():
                if not any(no_drop_name in n for no_drop_name in no_dropback_list) and layer_num>self.skip_init_layers:
                    list_names_dropback.append(n)
                    list_dropback.append(p)
                else:
                    list_names_ndropback.append(n)
                    list_no_dropback.append(p)
                layer_num=layer_num+1
            params = [
                {
                    "params": list_dropback,
                    "weight_decay": weight_decay,
                    "dropback":True,
                },
                {
                    "params": list_no_dropback,
                    "weight_decay": weight_decay,
                    "dropback":False,
                },
            ]                   
            self.actual_drop_rate       = []
            self.dropback_rate_channels = 0
        super(SGD_Dropback, self).__init__(params, defaults)
        
    

    def __setstate__(self, state):
        super(SGD_Dropback, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    # get current average and zero out the array
    def _get_dropback_prob_mean_(self,reset=True):
        if self.actual_drop_rate==[]:
            if self.dropback_rate!=0:
                print("Warning - empty drop_mean")
            return 0
        actual_drop_mean=np.mean(torch.FloatTensor(self.actual_drop_rate).cpu().detach().numpy())
        if reset is True:
            self.actual_drop_rate=[]
        return actual_drop_mean
        
    def _get_dropback_rate_(self):
        return self.dropback_rate

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                # sample layer with a prob dropback_rate, at least input output
                if group['dropback']==True and len(p.grad.shape)>=2: 
                    dropped_rate_keep=1 # how many to keep
                    # sample layer
                    if torch.rand(1)[0]<self.dropback_rate: 
                        # sample channel
                        dropped_mask                = (torch.rand(p.grad.shape[1]) < self.dropback_rate_channels).to(p.grad.device)
                        dropped_rate_keep           = (dropped_mask.shape.numel()-dropped_mask.sum())/dropped_mask.shape.numel()                        
                        # not likely but can be, in this case we do not drop anything
                        if dropped_rate_keep<=0: 
                            print("drop ALL!!!!",dropped_rate_keep)
                            dropped_rate_keep = 1
                        else:
                            p.grad[:,dropped_mask,...]= 0.0
                    d_p = p.grad/dropped_rate_keep
                    self.actual_drop_rate.append(1-dropped_rate_keep)
                else:                                
                    d_p = p.grad

                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-group['lr'])

        return loss