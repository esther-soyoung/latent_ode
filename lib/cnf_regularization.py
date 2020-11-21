import torch
import torch.nn as nn



class RegularizedODEfunc(nn.Module):
    def __init__(self, odefunc, regfunc, device):
        super(RegularizedODEfunc, self).__init__()
        self.odefunc = odefunc
        self.regfunc = regfunc
        self.device = device

    def before_odeint(self, *args, **kwargs):
        self.odefunc.before_odeint(*args, **kwargs)

    def forward(self, t, state):
        with torch.enable_grad():
            # x, logp = state[:2]
            x = state[0]
            x.requires_grad_(True)
            t.requires_grad_(True)
            # logp.requires_grad_(True)
            dstate = self.odefunc(t, x)
            if len(state) > 1:  # reg state
                reg_state = self.regfunc(dstate, t, self.device)
                return (dstate, reg_state)  # [3, 50, 20], [3]
            else:
                return dstate

    @property
    def get_nfe(self):
        return self.odefunc.get_nfe()

    def reset_nfe(self):
        self.odefunc.reset_nfe()


def total_derivative(x, t, logp, dx, dlogp, unused_context):
    del logp, dlogp, unused_context

    directional_dx = torch.autograd.grad(dx, x, dx, create_graph=True)[0]

    try:
        u = torch.full_like(dx, 1/x.numel(), requires_grad=True)
        tmp = torch.autograd.grad( (u*dx).sum(), t, create_graph=True)[0]
        partial_dt = torch.autograd.grad(tmp.sum(), u, create_graph=True)[0]

        total_deriv = directional_dx + partial_dt
    except RuntimeError as e:
        if 'One of the differentiated Tensors' in e.__str__():
            raise RuntimeError('No partial derivative with respect to time. Use mathematically equivalent "directional_derivative" regularizer instead')

    tdv2 = total_deriv.pow(2).view(x.size(0), -1)

    return 0.5*tdv2.mean(dim=-1)

def directional_derivative(x, t, logp, dx, dlogp, unused_context):
    del t, logp, dlogp, unused_context

    directional_dx = torch.autograd.grad(dx, x, dx, create_graph=True)[0]
    ddx2 = directional_dx.pow(2).view(x.size(0),-1)

    return 0.5*ddx2.mean(dim=-1)

def quadratic_cost(dx, t, unused_context):
    del t
    # del x, logp, dlogp, t, unused_context
    dx = dx.view(dx.shape[0], -1)
    return 0.5 * dx.pow(2).mean(dim=-1)

# computation graph maintain해서 시도해보기
def highorder_derivative(dstate, t, device):
    del t
    dx, dlogp = dstate[:2]
    one = torch.ones(3, 50, 20, device=device) #, requires_grad=True

    order_dx = torch.autograd.grad(dx, x, dx, create_graph=True)[0]
    # order_dx = torch.autograd.grad(dx, t, one, create_graph=True)[0]
    highorder_dx = torch.autograd.grad(order_dx, x, order_dx)[0]
    highorder_dx2 = highorder_dx.pow(2).view(x.size(0),-1)

    return highorder_dx

def jacobian_frobenius_regularization_fn(x, t, logp, dx, dlogp, context):
    sh = x.shape
    del logp, dlogp, t, dx, x
    sqjac = context.sqjacnorm

    return context.sqjacnorm
