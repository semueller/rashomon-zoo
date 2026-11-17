from __future__ import annotations

import logging
from typing import Callable




import numpy as np




import torch




from torch.utils.data import TensorDataset, DataLoader


from captum.attr import KernelShap, Lime




from .datasets import TorchRandomSeed
from .util import _get_targets


def _calc_gradients(model, data, inference_fn=None, device='cpu', return_outputs=False):
    """
    calculates gradients of data wrt. labels for inference_fn on device
    :param inference_fn: callable that returns full model output for a datum
    :param data: iterable containing the data
    :param device: device to where data is loaded
    :return:
    """
    if inference_fn is None:
        inference_fn = model
    model.zero_grad()
    gradients = []
    if return_outputs:
        _outputs = []
    try:
        with torch.set_grad_enabled(True):
            for i, (x, y) in enumerate(data):
                x = x.to(device).requires_grad_()
                y = y.to(device)
                outputs = inference_fn(x)
                # _selected_outputs = outputs[:, y]  # since y is a tensor and not a scalar this doesn't work
                # see: e=torch.eye(10); y=torch.arange(10); torch.gather(e, 1, y.unsqueeze(1)).squeeze() return vector of just 1.'s
                _selected_outputs = torch.gather(outputs, 1, y.unsqueeze(1)).squeeze()
                grads = torch.autograd.grad(torch.unbind(_selected_outputs), x)
                gradients.append(grads[0].to('cpu').detach())
                if return_outputs:
                    _outputs.append(outputs.to('cpu').detach())

        if return_outputs:
            return torch.vstack(gradients), torch.vstack(_outputs)
        else:
            return torch.vstack(gradients)

    except RuntimeError as re:
        if "CUDA out of memory" in str(re):
            logging.warning(f"attribution._calc_gradients: CUDA out of memory. Retrying on CPU.")
            model.to('cpu')  # this is why we need the model
            # value of return_outputs doesn't matter for the return syntax
            return_value = _calc_gradients(model, data, inference_fn=inference_fn,
                                           device='cpu', return_outputs=return_outputs)
            logging.warning(f"attribution._calc_gradients: Loading model back to GPU.")
            model.to('cuda')  # putting model back
            return return_value
        else:
            raise re


def vanilla_grad(model: torch.nn.Module, data: torch.Tensor, targets: torch.Tensor | None = None,
                 inference_fn: Callable | None = None, device=None, simplified: bool = False,
                 pre_process_fn: Callable | None = None) -> torch.Tensor | tuple:
    return smooth_grad(inference_fn=inference_fn, model=model, data=data, targets=targets,
                       n_samples=1, std=0., random_state=0, device=device, simplified=simplified,
                       pre_process_fn=pre_process_fn)


def smooth_grad(model: torch.nn.Module, data: torch.Tensor, targets: torch.Tensor | None = None,
                inference_fn: Callable | None = None, std: float = 1., n_samples: int = 50, random_state: int = 42,
                device=None, simplified: bool = False, noise_level: float = 0.1, pre_process_fn: Callable | None = None,
                force_random: bool = True) -> torch.Tensor | tuple:
    # expecting targets.shape=(n,)

    if device is None:
        device = next(model.parameters()).device

    if inference_fn is None:
        inference_fn = model


    if pre_process_fn is not None:
        model = model.to('cpu')
        data = data.to('cpu')
        data = pre_process_fn(data)
        model = model.to(device)

    targets_was_none = targets is None
    if targets is None:
        targets = _get_targets(inference_fn, data, model, device)

    noise = noise_level * std
    # independent noise for every sample and copy (as in SmoothGrad); the legacy code drew one set of n_samples
    # vectors and reused it for all samples, which correlated the errors across samples (changed 2026-11-17)
    n_draws = data.shape[0] * n_samples if n_samples > 1 else n_samples
    if force_random:
        perturbation = torch.normal(mean=0., std=noise, size=(n_draws, *data.shape[1:]))
    else:
        with TorchRandomSeed(random_state):
            perturbation = torch.normal(mean=0., std=noise, size=(n_draws, *data.shape[1:]))

    if n_samples <= 1:
        perturbed_data = data
        targets_expanded = targets
        batch_size = data.shape[0]
    else:
        # each element in dim 0 is repeated n_samples times, contiguously
        perturbed_data = torch.repeat_interleave(data, repeats=n_samples, dim=0)
        # same order as perturbed_data (sample-major). The legacy code (lxg, RashomonEffect) used
        # targets.unsqueeze(0).expand(n_samples, n).reshape(-1), i.e. noise-major: rows got the targets of other
        # samples whenever a batch held more than one class (fixed 2026-11-17)
        targets_expanded = torch.repeat_interleave(targets, repeats=n_samples, dim=0)
        batch_size = n_samples

    perturbed_data = perturbed_data + perturbation
    _data = DataLoader(TensorDataset(perturbed_data, targets_expanded),
                       shuffle=False, batch_size=batch_size)

    _prev_train_state = model.training
    model.train(True)
    _grads = _calc_gradients(model=model, data=_data, inference_fn=inference_fn, device=device)
    model.train(_prev_train_state)

    # sum everything up and out they go!
    idxs = torch.arange(0, perturbed_data.shape[0]+n_samples, n_samples)
    gradients = []
    for a, b in zip(idxs[:-1], idxs[1:]):
        gradients.append(
            torch.mean(_grads[a:b], dim=0)
        )

    gradients = torch.stack(gradients)

    if simplified:
        gradients = torch.sum(gradients, dim=-1)

    if targets_was_none:
        return (gradients, targets)

    return gradients


def integrated_gradients(model: torch.nn.Module, data: torch.Tensor, targets: torch.Tensor | None = None,
                         inference_fn: Callable | None = None, baselines: torch.Tensor | None = None,
                         n_samples: int = 100, simplified: bool = False, calc_paths: Callable | None = None,
                         calc_baselines: Callable | None = None, fit_baseline_data: bool = False, device=None,
                         outputmode: str | None = None, pre_process_fn: Callable | None = None,
                         return_convergence_delta: bool = False, _batch_size: int = 256,
                         subtract_baseline: bool = False) -> torch.Tensor | tuple:
    """
    :param model: Classifier model
    :param data: input data, batch first
    :param targets: target class to compute attribution for, if None, inference is run on samples once and predicted class is chosen
    :param inference_fn: optional, function used to run inference with. If None then inference_fn=model
    :param baselines: if none given, zero baseline is used,
        "integral approximated with left riemannian integration plus inclusion of final point",
        ie we interpolate linearly , x_0 + t(x - x_0) t = [0., ..., 1.], and include start and end points
    :param n_samples: number of steps between baseline and input -> actual steps = steps+2; only used when calc_paths is not a callable
    :param simplified: return attribution of shape (1, seqlen), accumulate attribution for each vector
            if false, returns all collected gradients of size [n_samples, n_steps, seqlen, embedding_size]
    :param calc_paths: needs to return paths between all samples as well as distance between consecutive samples
    :param calc_baselines: must return one baseline per sample
    :return:
    """

    if device is None:
        device = next(model.parameters()).device  #  this looks ugly but is apparently the way to go in vanilla pytorch

    if inference_fn is None:
        inference_fn = model

    _use_tuple = isinstance(data, tuple)

    # data = data[:2]
    # targets = targets[:2]

    if pre_process_fn is not None:
        model = model.to('cpu')
        data = data.to('cpu')
        data = pre_process_fn(data)
        model = model.to(device)

    ## put everything on cpu for now
    if _use_tuple:
        data = data[0].to('cpu')
    else:
        data = data.to('cpu')

    input = data

    # support multiple baselines per sample?
    if callable(calc_baselines):
        baselines = calc_baselines(data)  # (samples, model)?
    if baselines is None:
        baselines = torch.zeros_like(input)
    elif fit_baseline_data:
        if len(baselines.shape) == 1:
            baselines = np.expand_dims(baselines, axis=0)
        baselines = np.repeat(baselines, len(input), 0)


    # paths.size = [n_samples, n_steps, seqlen, embedding]
    paths, scaling = None, None
    if callable(calc_paths):
        paths, scaling = calc_paths(baselines, input, n_samples)
    else:
        # fallback: straight line between baseline and input,
        p = np.linspace(baselines, input, num=n_samples, dtype=np.float32)#, retstep=True)
        paths = torch.tensor(p).swapaxes(0, 1).requires_grad_()
        scaling = 1. / n_samples
        # paths = torch.tensor(np.linspace(baselines, samples, num=steps)).swapaxes(0,1).requires_grad_()
        # paths = paths[:, :-1, :]

    assert scaling is not None and paths is not None

    targets_was_none = targets is None
    if targets is None:
        targets = _get_targets(inference_fn, data, model, device)


    output_shape = inference_fn(paths[0].to(device)).shape
    # places gradients in attribution
    _prev_train_state = model.training
    model.train(True)
    attribution = torch.zeros_like(paths)
    outputs = torch.zeros(*paths.shape[:2], output_shape[1])
    for i in range(paths.shape[0]):
        _path_one_sample = paths[i]
        _target_output = targets[i].expand(paths.shape[1])
        _dataset = DataLoader(TensorDataset(_path_one_sample, _target_output), shuffle=False, batch_size=_batch_size)
        attribution[i], outputs[i] = _calc_gradients(model=model, data=_dataset,
                                      inference_fn=inference_fn, device=device, return_outputs=True)
    model.train(_prev_train_state)


    # from captum.attr import IntegratedGradients
    # explainer = IntegratedGradients(inference_fn,  multiply_by_inputs=False)
    # _attr2, delta = explainer.attribute(input, target=targets,
    #                                     n_steps=steps, return_convergence_delta=True)
    #
    # _attr_eq = torch.mean(attribution, dim=1)

    # scale gradients according to distance between point and previous point
    if hasattr(scaling, 'shape') and scaling.shape == paths.shape[:-1]:
        for _ in range(paths.shape[0]):
            attribution[0] = attribution[0] * scaling[0].unsqueeze(-1)
    else:
        if np.isscalar(scaling):
            scaling = torch.full((n_samples,), scaling)
        if scaling.shape[:2] != (input.shape[0], n_samples):
            for _ in range(len(paths.shape[2:])):
                scaling = scaling.unsqueeze(-1)
            for i in range(attribution.shape[0]):
                attribution[i] *= scaling

    # sum up over path
    attribution = torch.sum(attribution, dim=1)

    if simplified:
        attribution = torch.sum(attribution, dim=-1)

    if _use_tuple:
        attribution = (attribution,)  # OMG DUH GRHNG

    _numerical_delta = torch.nan
    if outputmode == 'full' or return_convergence_delta:

        at_baseline = torch.gather(outputs[:, 0], -1, targets.unsqueeze(1)).squeeze()
        at_sample = torch.gather(outputs[:, -1], -1, targets.unsqueeze(1)).squeeze()
        diff = at_sample - at_baseline
        # for tabular data -> obtain attribution mass of each sample
        _global_attrs = attribution * (data - baselines)
        _sums = torch.sum(_global_attrs, -1)
        # for non-tabular data we need to keep summing
        while len(_sums.shape) > 1:
            _sums = torch.sum(_sums, -1)
        _numerical_delta = torch.abs(diff - _sums)
        # perc = diff / _sums
        # _delta_percent = (1.0 - perc.abs()).abs()
        # print(torch.sum(torch.le(_delta_percent, 0.05)))
        # assert torch.all(torch.le( _delta_percent, 0.05))  # check if percentage of difference is less than 5% for all samples

    if subtract_baseline:
        attribution = attribution * (data - baselines)

    if outputmode == 'full':
        if targets_was_none:
            return attribution, paths, scaling, _numerical_delta, targets
        return attribution, paths, scaling, _numerical_delta


    if return_convergence_delta:
        if targets_was_none:
            return attribution, _numerical_delta, targets
        return attribution, _numerical_delta

    if targets_was_none:
        return attribution, targets

    return attribution


def kernelshap(model: torch.nn.Module, data: torch.Tensor, targets: torch.Tensor | None = None,
               inference_fn: Callable | None = None, baselines: torch.Tensor | None = None, device=None,
               n_samples: int | None = None, masks: torch.Tensor | None = None,
               baseline: torch.Tensor | None = None) -> torch.Tensor | tuple:
    """
    **tested for tabular data only so far**
    uses lime with appropriate loss function (MSE), weighting kernel (Shapley Kernel) and regularization (ie none)
    :param model: model
    :param data: DataLoader
    :param targets: target class to compute attribution for, if None, inference is run on samples once and predicted class is chosen
    :param inference_fn: optional, function used to run inference with. If None then inference_fn=model
    :param baselines: defaultis None; scalar, single baseline or one baseline per input
    :param n_samples: default 25; number of samples to train regression on
    :return: Attribution scores computed by KernelShap

    """

    if device is not None:
        model = model.to(device)

    if device is None:
        device = next(model.parameters()).device

    if inference_fn is None:
        inference_fn = model

    if n_samples is None:
        n_samples = 2 * data[0].shape[0]

    # attributions = []
    # baselines = _check_baselines(baselines)
    # if masks is None:
    #     masks = [None]
    # for x, b, y, m in zip(data, cycle(baselines), targets, cycle(masks)):
    explainer = KernelShap(inference_fn)
    data = data.to(device)

    targets_was_none = targets is None
    if targets is None:
        targets = _get_targets(inference_fn, data, model, device)

    if targets is not None:
        targets = targets.to(device)
    if masks is not None:
        masks = masks.to(device)
    _prev_train_state = model.training
    model.train(True)
    attributions = explainer.attribute(inputs=data, baselines=baselines, target=targets, n_samples=n_samples,
                                       feature_mask=masks, perturbations_per_eval=2048)
    model.train(_prev_train_state)

    if targets_was_none:
        return attributions, targets
    return attributions


def lime(model: torch.nn.Module, data: torch.Tensor, targets: torch.Tensor | None = None,
         inference_fn: Callable | None = None, baselines: torch.Tensor | None = None, device=None,
         n_samples: int = 25, distance_mode: str = "cosine") -> torch.Tensor | tuple:
    """
    **tested for tabular data only so far**

    :param model: model
    :param data: input data, expects first dim to be batchsize
    :param targets: target class to compute attribution for, if None, inference is run on samples once and predicted class is chosen
    :param inference_fn: optional, function used to run inference with. If None then inference_fn=model
    :param baselines: defaultis None; scalar, single baseline or one baseline per input
    :param n_samples: default 25; number of samples to train regression on
    :param distance_mode: default "cosine"; distance mode to be used in loss for surrogate model, alternatively "euclidean"
    :return: Attribution scores computed by Lime
    """

    if device is not None:
        model = model.to(device)

    if device is None:
        device = next(model.parameters()).device

    if inference_fn is None:
        inference_fn = model

    targets_was_none = targets is None
    if targets is None:
        targets = _get_targets(inference_fn, data, model, device)


    from captum.attr._core.lime import get_exp_kernel_similarity_function
    similarity_fn = get_exp_kernel_similarity_function(distance_mode=distance_mode)

    explainer = Lime(inference_fn, similarity_func=similarity_fn)

    data = data.to(device)
    if targets is not None:
        targets = targets.to(device)
    _prev_train_state = model.training
    model.train(True)
    attributions = explainer.attribute(inputs=data, baselines=baselines, target=targets, n_samples=n_samples,
                                       perturbations_per_eval=2048)  # n_samples=25 provided the same ranking for top 10 features as n=250
    model.train(_prev_train_state)

    if targets_was_none:
        return attributions, targets
    return attributions


