# initium

Libraries:
pip install git+https://github.com/ixaxaar/pytorch-dnc.git

import torch, re
torch_ver = torch.__version__.split("+")[0]                 # e.g. "2.10.0"
torch_mm  = ".".join(torch_ver.split(".")[:2])               # e.g. "2.10"
cuda_ver  = torch.version.cuda                               # e.g. "12.8"
cuda_tag  = "cu" + cuda_ver.replace(".", "")                 # e.g. "cu128"
print(f"torch={torch_ver}  torch_mm={torch_mm}  cuda={cuda_ver}  cuda_tag={cuda_tag}")

index_url = f"https://wheels.astral.sh/simple/{cuda_tag}/"

# mamba-ssm: pull from Astral's GPU index, ADDING it alongside PyPI
# (--extra-index-url, not --index-url) so torch/einops/etc still resolve
# normally and only mamba-ssm itself comes from the matched-wheel index.
pip install "mamba-ssm==2.3.2.post1+cu.{cuda_ver}.torch.{torch_mm}" \
    --extra-index-url {index_url}
