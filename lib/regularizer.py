
''' kinetic energy term
'''
def quadratic_cost(dx):
    dx = dx.view(dx.shape[0], -1)
    return dx.pow(2).mean(dim=-1)

''' higher order derivatives
'''
