import torch
import numpy as np
from .utils import * #group_product, group_add, normalization, get_params_grad, hessian_vector_product, orthnormal, l2


class hessian():

    def __init__(self, model, criterion, data=None, dataloader=None, cuda=True, weight_decay=1e-4):
        assert (data is not None and dataloader is None) or (data is None and dataloader is not None)

        self.model = model.eval()  # make model is in evaluation model
        self.criterion = criterion

        if data is not None:
            self.data = data
            self.full_dataset = False
        else:
            self.data = dataloader
            self.full_dataset = True

        self.device = torch.device('cuda' if cuda else 'cpu')
        self.weight_decay = weight_decay

        # pre-processing for single batch case to simplify the computation.
        if not self.full_dataset:
            
            if len(self.data) == 2:
                self.inputs2, self.targets = self.data
                if self.device.type == 'cuda':
                    # self.targets = self.targets.long().to(self.device)
                    self.inputs2, self.targets = self.inputs2.to(self.device), self.targets.long().to(self.device)
                
                # if we only compute the Hessian information for a single batch data, we can re-use the gradients.
                outputs = self.model(self.inputs2)
                loss = self.criterion(outputs, self.targets)# + self.weight_decay * l2(self.model, self.device == 'cuda')

                
            if len(self.data) == 3:
                self.inputs1, self.inputs2, self.targets = self.data
                if self.device.type == 'cuda':
                    self.inputs1, self.targets = self.inputs1.float().to(self.device), self.targets.long().to(self.device)
                    if isinstance(self.inputs2, list):
                       pass
                    else:
                        self.inputs2 = self.inputs2.to(self.device)
                
                 # if we only compute the Hessian information for a single batch data, we can re-use the gradients.
                outputs = self.model(self.inputs1, self.inputs2)

                loss = self.criterion(outputs, self.targets)# + self.weight_decay * l2(self.model, self.device == 'cuda')

                    
            if len(self.data) == 4:
                text, audio, video, labels_list = self.data
                labels, modality = labels_list
                if self.device.type == 'cuda':
                    text, audio, labels = text.float().to(self.device), audio.float().to(self.device), labels.long().to(self.device)
                
                # with torch.backends.cudnn.flags(enabled=False):
                if modality == 'audio':
                    outputs = self.model(text, audio, pad_x=True, pad_y=False)
                elif modality == 'text':
                    outputs = self.model(text, audio, pad_x=False, pad_y=True)
                else:
                    outputs = self.model(text, audio, pad_x=False, pad_y=False)
                    
                loss = self.criterion(outputs, labels) + self.weight_decay * l2(self.model, self.device.type == 'cuda')
               
            if len(self.data) == 5:
                self.inputs1, self.inputs2, self.inputs3, self.input4, self.targets = self.data
                if self.device.type == 'cuda':
                    self.inputs1, self.inputs2, self.inputs3, self.targets = self.inputs1.float().to(self.device), self.inputs2.float().to(self.device), self.inputs3.float().to(self.device), self.targets.long().to(self.device)

                outputs = self.model(self.inputs1, self.inputs2, self.inputs3, self.input4)
                loss = self.criterion(outputs, self.targets) + self.weight_decay * l2(self.model, self.device.type == 'cuda')
                # grads = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
                
            loss.backward(create_graph=True)  # compute the gradients

        # this step is used to extract the parameters from the model
        params, gradsH = get_params_grad(self.model)
        self.params = params
        self.gradsH = gradsH  # gradient used for Hessian computation

    def _move_to_device(self, obj):
        if torch.is_tensor(obj):
            return obj.to(self.device)
        if isinstance(obj, dict):
            return {k: self._move_to_device(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._move_to_device(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._move_to_device(v) for v in obj)
        return obj

    def _first_tensor(self, obj):
        if torch.is_tensor(obj):
            return obj
        if isinstance(obj, dict):
            for v in obj.values():
                t = self._first_tensor(v)
                if t is not None:
                    return t
            return None
        if isinstance(obj, (list, tuple)):
            for v in obj:
                t = self._first_tensor(v)
                if t is not None:
                    return t
            return None
        return None

    def dataloader_hv_product(self, v):
        """
        Compute (H_full v) where H_full is the Hessian of the *dataset mean* objective.
        We do:
            THv = (1/N) * sum_batches (b * (H_batch_mean v))
        """
        THv = [torch.zeros_like(p, device=self.device) for p in self.params]
        num_samples = 0.0

        for batch in self.data:
            if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                raise ValueError("Each dataloader batch must be a tuple/list with model inputs and targets.")

            *inputs, targets = batch
            if len(inputs) == 0:
                raise ValueError("Dataloader batch is missing model inputs.")

            inputs = [self._move_to_device(inp) for inp in inputs]
            targets = self._move_to_device(targets)
            if torch.is_tensor(targets):
                targets = targets.long()

            target_tensor = self._first_tensor(targets)
            if target_tensor is not None and target_tensor.dim() > 0:
                b = float(target_tensor.size(0))
            else:
                input_tensor = self._first_tensor(inputs)
                if input_tensor is None or input_tensor.dim() == 0:
                    raise ValueError("Unable to infer batch size from dataloader batch.")
                b = float(input_tensor.size(0))

            self.model.zero_grad(set_to_none=True)

            if len(inputs) == 1:
                outputs = self.model(inputs[0])
            else:
                outputs = self.model(*inputs)

            # --- per-sample mean data loss (batch-size invariant) ---
            loss_data = self.criterion(outputs, targets)
            if loss_data.dim() > 0:
                loss_data = loss_data.mean()
            else:
                red = getattr(self.criterion, "reduction", "mean")
                if red == "sum":
                    loss_data = loss_data / b

            loss = loss_data + self.weight_decay * l2(self.model, self.device.type == "cuda")

            grads = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
            for p, g in zip(self.model.parameters(), grads):
                p.grad = g

            params, gradsH = get_params_grad(self.model)

            self.model.zero_grad(set_to_none=True)

            Hv = torch.autograd.grad(
                gradsH, params,
                grad_outputs=v,
                only_inputs=True,
                retain_graph=False
            )

            # sample-weighted average across batches: sum(b * Hv_batch) / sum(b)
            THv = [THv_i + Hv_i * b for THv_i, Hv_i in zip(THv, Hv)]
            num_samples += b

        THv = [THv_i / num_samples for THv_i in THv]
        eigenvalue = group_product(THv, v).detach().cpu().item()
        return eigenvalue, THv

    
    def eigenvalues(self, maxIter=100, tol=1e-3, top_n=1):
        """
        compute the top_n eigenvalues using power iteration method
        maxIter: maximum iterations used to compute each single eigenvalue
        tol: the relative tolerance between two consecutive eigenvalue computations from power iteration
        top_n: top top_n eigenvalues will be computed
        """

        assert top_n >= 1

        device = self.device

        eigenvalues = []
        eigenvectors = []

        computed_dim = 0

        while computed_dim < top_n:
            eigenvalue = None
            v = [torch.randn(p.size()).to(device) for p in self.params]  # generate random vector
            v = normalization(v)  # normalize the vector

            for i in range(maxIter):
                v = orthnormal(v, eigenvectors)
                self.model.zero_grad()

                if self.full_dataset:
                    tmp_eigenvalue, Hv = self.dataloader_hv_product(v)
                else:
                    Hv = hessian_vector_product(self.gradsH, self.params, v)
                    tmp_eigenvalue = group_product(Hv, v).cpu().item()

                v = normalization(Hv)

                if eigenvalue == None:
                    eigenvalue = tmp_eigenvalue
                else:
                    if abs(eigenvalue - tmp_eigenvalue) / (abs(eigenvalue) + 1e-6) < tol:
                        break
                    else:
                        eigenvalue = tmp_eigenvalue
            eigenvalues.append(eigenvalue)
            eigenvectors.append(v)
            computed_dim += 1

        return eigenvalues, eigenvectors

    def trace(self, maxIter=100, tol=1e-3):
        """
        compute the trace of hessian using Hutchinson's method
        maxIter: maximum iterations used to compute trace
        tol: the relative tolerance
        """
        device = self.device
        trace_vhv = []
        trace = 0.

        for i in range(maxIter):
            self.model.zero_grad()
            v = [torch.randint_like(p, high=2, device=device) for p in self.params]
            for v_i in v:  # generate Rademacher random variables
                v_i[v_i == 0] = -1

            _, Hv = self.dataloader_hv_product(v) if self.full_dataset else (None, hessian_vector_product(self.gradsH, self.params, v))
            trace_vhv.append(group_product(Hv, v).cpu().item())

            if abs(np.mean(trace_vhv) - trace) / (abs(trace) + 1e-9) < tol:
                return trace_vhv
            trace = np.mean(trace_vhv)

        return trace_vhv

    def density(self, iter=100, n_v=1):
        """
        compute estimated eigenvalue density using stochastic lanczos algorithm (SLQ)
        iter: number of iterations used to compute trace
        n_v: number of SLQ runs
        """
        device = self.device
        eigen_list_full, weight_list_full = [], []

        for k in range(n_v):
            v = [torch.randint_like(p, high=2, device=device) for p in self.params]
            for v_i in v:  # generate Rademacher random variables
                v_i[v_i == 0] = -1
            v = normalization(v)

            # standard lanczos algorithm initialization
            v_list, w_list, alpha_list, beta_list = [v], [], [], []

            ############### Lanczos
            for i in range(iter):
                self.model.zero_grad()
                w_prime = [torch.zeros(p.size()).to(device) for p in self.params]
                
                if i == 0:
                    _, w_prime = self.dataloader_hv_product(v) if self.full_dataset else (None, hessian_vector_product(self.gradsH, self.params, v))
                    alpha = group_product(w_prime, v)
                    alpha_list.append(alpha.cpu().item())
                    w = group_add(w_prime, v, alpha=-alpha)
                    w_list.append(w)
                else:
                    beta = torch.sqrt(group_product(w, w))
                    beta_list.append(beta.cpu().item())
                    v = orthnormal(w, v_list) if beta_list[-1] != 0. else orthnormal([torch.randn(p.size()).to(device) for p in self.params], v_list)
                    v_list.append(v)
                    _, w_prime = self.dataloader_hv_product(v) if self.full_dataset else (None, hessian_vector_product(self.gradsH, self.params, v))
                    alpha = group_product(w_prime, v)
                    alpha_list.append(alpha.cpu().item())
                    w_tmp = group_add(w_prime, v, alpha=-alpha)
                    w = group_add(w_tmp, v_list[-2], alpha=-beta)

            T = torch.zeros(iter, iter).to(device)
            for i in range(len(alpha_list)):
                T[i, i] = alpha_list[i]
                if i < len(alpha_list) - 1:
                    T[i + 1, i] = beta_list[i]
                    T[i, i + 1] = beta_list[i]

            eigenvalues, eigenvectors = torch.linalg.eig(T)
            eigen_list_full.append(list(eigenvalues.real.cpu().numpy()))
            weight_list_full.append(list(torch.pow(eigenvectors[0, :], 2).cpu().numpy()))

        return eigen_list_full, weight_list_full

