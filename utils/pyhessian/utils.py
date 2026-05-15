import torch

def group_product(xs, ys):
    """
    the inner product of two lists of variables xs,ys
    :param xs:
    :param ys:
    :return:
    """
    return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])


def group_add(params, update, alpha=1):
    """
    params = params + update*alpha
    :param params: list of variable
    :param update: list of data
    :return:
    """
    for i, p in enumerate(params):
        params[i].data.add_(update[i] * alpha)
    return params

def l2(model, gpu=False):
    l2_norm = torch.tensor(0.)
    l2_norm = l2_norm.cuda() if gpu else l2_norm

    for param in model.parameters():
        l2_norm += torch.sum(param**2)

    return l2_norm / 2

# def l2(model, gpu=False):
#     device = "cuda" if gpu else "cpu"
#     return torch.linalg.norm(torch.cat([p.flatten() for p in model.parameters()])).to(device)

def normalization(v):
    """
    normalization of a list of vectors
    return: normalized vectors v
    """
    s = group_product(v, v)
    s = s**0.5
    s = s.cpu().item()
    v = [vi / (s + 1e-9) for vi in v]
    return v


def get_params_grad(model):
    """
    get model parameters and corresponding gradients
    """
    params = []
    grads = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        params.append(param)
        grads.append(0. if param.grad is None else param.grad + 0.)
    return params, grads


def hessian_vector_product(gradsH, params, v):
    """
    compute the hessian vector product of Hv, where
    gradsH is the gradient at the current point,
    params is the corresponding variables,
    v is the vector.
    """
    hv = torch.autograd.grad(gradsH,
                             params,
                             grad_outputs=v,
                             only_inputs=True,
                             retain_graph=True)
    return hv


def orthnormal(w, v_list):
    """
    make vector w orthogonal to each vector in v_list.
    afterwards, normalize the output w
    """
    for v in v_list:
        w = group_add(w, v, alpha=-group_product(w, v))
    return normalization(w)


def efficient_hessian_vector_product(loss, params, v, retain_graph=True):
    """
    Computes the Hessian-vector product Hv where H is the Hessian of the loss with respect to params,
    and v is a list of vectors with the same shape as params.
    Uses the Hutchinson's trick to avoid computing the full Hessian.
    
    Args:
        loss: Scalar tensor from which to compute gradients
        params: List of parameters to compute the Hessian with respect to
        v: List of vectors with the same shape as params
        retain_graph: Whether to retain the computation graph
        
    Returns:
        List of tensors containing the Hessian-vector product
    """
    # First-order gradient
    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
    
    # Compute the product of gradients with the vectors v
    grad_vector_product = sum(torch.sum(g * v_i) for g, v_i in zip(grads, v))
    
    # Compute the Hessian-vector product using a single backward pass
    hvp = torch.autograd.grad(grad_vector_product, params, retain_graph=retain_graph)
    
    return hvp