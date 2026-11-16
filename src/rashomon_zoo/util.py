

import pathlib


import torch




import numpy as np


import pickle as pkl


import json


from pathlib import Path






from torch.utils.data import DataLoader, TensorDataset










class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)):
            return None

        return json.JSONEncoder.default(self, obj)


def create_checkpoint(path, model, optimizer=None):
    if optimizer is None:
        torch.save({
            'model_state_dict': model.state_dict(),
        }, path)
    else:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, path)


def restore_checkpoint(path, model, optimizer=None, train=True):
    if not torch.cuda.is_available():
        ckpt = torch.load(path, map_location=torch.device('cpu'))
    else:
        ckpt = torch.load(path, map_location=torch.device('cuda'))
    for k, v in ckpt['model_state_dict'].items():
        if not v.is_contiguous():
            ckpt['model_state_dict'][k] = v.contiguous()
    model.load_state_dict(ckpt['model_state_dict'])
    if train:
        model.train()
    try:
        if optimizer is not None:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    except KeyError as ke:
        print(str(ke))
        print("RESTORE_CHECKPOINT: Optimizer given but no Optimizer found in state dict.")


def _get_filenames(pth: Path):
    return [
        str(f) for f in pth.iterdir() if f.is_file()
    ]


def load_meta_data(data_dir, just_one=False):
    fnames = _get_filenames(Path(data_dir))
    meta_data = {}
    for fname in fnames:
        if 'meta_data' in fname:
            #print(f'loading {fname}')
            dataseed = fname.split('_')[-1].split('.')[0]
            with open(fname, 'rb') as md:
                data = pkl.load(md)
                if just_one:
                    return data
                else:
                    meta_data[dataseed] = data

    return meta_data


def _load_meta_data_by_pid(data_dir):
    fnames = _get_filenames(Path(data_dir))
    meta_data = {}
    for fname in fnames:
        if 'meta_data' in fname:
            #print(f'loading {fname}')
            pid = fname.split('_')[2]
            with open(fname, 'rb') as md:
                data = pkl.load(md)
                meta_data[pid] = data
    return meta_data


def dump_pkl(data, fname, mode='pkl'):
    if type(fname) == Path:
        fname = str(fname)
    if 'pkl' in mode:
        with open(fname, 'wb') as f:
            pkl.dump(data, f)
    elif 'json' in mode:
        with open(fname, "w") as f:
            json.dump(data, f, cls=NumpyEncoder)
    else:
        raise ValueError(f"dump_pkl: invalid value for mode. val: {mode}")


def load_pkl(fname, mode='pkl'):
    if type(fname) == Path or type(fname) == pathlib.PosixPath:
        fname = str(fname)
    if fname.endswith('.pkl'):
        with open(fname, 'rb') as f:
            data = pkl.load(f)
    elif fname.endswith('.json'):
        with open(fname, "r") as f:
            data = json.load(f)
    else:
        raise ValueError(f"load_pkl: invalid file type '{fname.split('.')[-1]}', must be pkl or json")
    return data


def _get_outputs(inference_fn, data, model, device, batch_size=512):

    _data = DataLoader(TensorDataset(data), shuffle=False, batch_size=batch_size)

    try:
        _y_out = []
        for x in _data:
            _y = inference_fn(x[0].to(device))
            _y_out.append(_y.cpu())
        return torch.vstack(_y_out)
    except RuntimeError as re:
        if "CUDA out of memory" in str(re):
            model.to('cpu')
            outputs = _get_outputs(inference_fn, data, model, 'cpu')
            model.to('cuda')
            return outputs
        else:
            raise re


def _get_targets(inference_fn, data, model, device):
    return torch.argmax(_get_outputs(inference_fn, data, model, device), dim=1)
