"""Actual CUDA arithmetic and read-only tensor integrity before study fitting."""
import torch


def check_cuda(device):
    """Exact integer-valued FP32 probes permit no numerical tolerance."""
    probe=torch.randn((64,64),device=device,requires_grad=True)
    with torch.autocast('cuda',dtype=torch.float16):value=(probe@probe.T).square().mean()
    value.backward();torch.cuda.synchronize(device)
    if not bool(torch.isfinite(value)) or not bool(torch.isfinite(probe.grad).all()):raise RuntimeError('Nonfinite mixed-precision matrix/gradient probe')
    del probe,value
    count=95*555*128
    reference=(torch.arange(count,dtype=torch.int64)%257).float().reshape(1,95,555,128)
    # This size exercises the column-encoder scale as well as simple GEMM.
    source=reference.to(device)
    for repetition in range(10):
        output=source+1.
        torch.cuda.synchronize(device)
        if not torch.equal(source.cpu(),reference):raise RuntimeError(f'CUDA read-only tensor integrity failed at repetition {repetition}; a source tensor changed without an update')
        if not torch.equal(output.cpu(),reference+1.):raise RuntimeError(f'CUDA exact arithmetic integrity failed at repetition {repetition}')
    del source,output,reference
    left=torch.randn((1,95,555,128),device=device)
    right=torch.randn_like(left)
    reference_left=left.cpu().clone();reference_right=right.cpu().clone()
    reference_sum=reference_left+reference_right
    for repetition in range(10):
        output=left+right
        torch.cuda.synchronize(device)
        if not torch.equal(left.cpu(),reference_left) or not torch.equal(right.cpu(),reference_right):
            raise RuntimeError(f'CUDA floating-point read-only tensor integrity failed at repetition {repetition}')
        if not torch.equal(output.cpu(),reference_sum):raise RuntimeError(f'CUDA floating-point addition integrity failed at repetition {repetition}')
    del left,right,output,reference_left,reference_right,reference_sum
    torch.cuda.empty_cache()
    return dict(status='passed',mixed_precision_backward_finite=True,read_only_tensor_elements=count,
        exact_fp32_repetitions=10,floating_point_pattern_repetitions=10,read_only_inputs_unchanged=True,exact_arithmetic=True)
