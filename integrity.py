"""Read-only transfer and repeated inference checks; no model transformation."""
import hashlib
import torch


def model_digest(model):
    digest=hashlib.sha256()
    for name,value in model.state_dict().items():
        value=value.detach().cpu().contiguous()
        digest.update(name.encode('utf-8'));digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode());digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def checked_transfer(inputs,device):
    reference=tuple(value.detach().cpu().clone() for value in inputs)
    transferred=tuple(value.to(device) for value in inputs)
    def verify():
        for index,(value,before) in enumerate(zip(transferred,reference)):
            if not torch.equal(value.detach().cpu(),before):
                raise RuntimeError(f'Read-only model input {index} changed during device computation; stop this run and inspect the compute environment')
    verify()
    return transferred,verify


def check_repeat(first,second,device):
    tolerance=5e-5 if device.type=='cuda' else 2e-6
    if not torch.isfinite(first).all() or not torch.isfinite(second).all():
        raise RuntimeError('Nonfinite repeated inference output')
    difference=float((first-second).abs().max())
    if difference>tolerance:
        raise RuntimeError(f'Unstable repeated inference with fixed inputs and weights: maximum difference {difference:.9g} exceeds {tolerance}; stop this run and inspect the compute environment')
    return difference


def check_model_unchanged(model,before):
    if model_digest(model)!=before:
        raise RuntimeError('Model parameters or buffers changed during read-only inference; stop this run and inspect the compute environment')
