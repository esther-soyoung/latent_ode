
''' kinetic energy term
'''
def quadratic_cost(dx):  # [3, 50, 2208, 41]
    dx = dx.view(dx.shape[1], -1)
    return dx.pow(2).mean(dim=-1)

''' higher order derivatives
'''
