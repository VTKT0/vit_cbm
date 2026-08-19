## this file makes some synthetic test data to model computational complexity

import logging
logger = logging.getLogger(__name__)

import os

log_file = 'batch.log'
if os.path.exists(log_file):
    os.remove(log_file)
with open(log_file, 'w'):
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),                 # prints to terminal
        logging.FileHandler(log_file)           # saves to file
    ]
)

import torch
import torch.nn as nn
import torch.nn.functional as F
# from huggingface_hub import login, hf_hub_download
import timm


def _log_gpu_mem(tag,x,model,optimizer):


    if not torch.cuda.is_available():
        return

    device = torch.cuda.current_device()

    allocated = torch.cuda.memory_allocated(device) 
    reserved = torch.cuda.memory_reserved(device) 
    peak = torch.cuda.max_memory_allocated(device) 

    if model is not None:
        param_bytes, grad_bytes = model_parameter_cost(model, mode='return')
    else:
        param_bytes = 0
        grad_bytes = 0

    if optimizer is not None:
        optimizer_bytes = optimizer_cost(optimizer, mode='return')
    else:
        optimizer_bytes = 0
    
    if x is not None:
        x_bytes = x.numel() * x.element_size()
    else:
        x_bytes = 0

    other_bytes = allocated - (param_bytes + grad_bytes + optimizer_bytes + x_bytes)

    table_width = 18

    tag_str = ('state=' + tag.upper()).ljust(25)
    allocated_str = f"allocated={allocated/1024**3:.2f}GB".ljust(table_width)
    reserved_str = f"reserved={reserved/1024**3:.2f}GB".ljust(table_width)
    peak_str = f"peak={peak/1024**3:.2f}GB".ljust(table_width)
    param_str = f"parameters={param_bytes/1024**3:.2f}GB".ljust(table_width)
    grad_str = f"gradients={grad_bytes/1024**3:.2f}GB".ljust(table_width)
    optimizer_str = f"optimizer={optimizer_bytes/1024**3:.2f}GB".ljust(table_width)    
    x_str = f"data={x_bytes/1024**3:.2f}GB".ljust(table_width)
    other_str = f"activations={other_bytes/1024**3:.2f}GB".ljust(table_width)
    logger.info(
        f"{tag_str} | "
        f"{allocated_str} | "
        f"{reserved_str} | "
        f"{peak_str} | "
        f"{param_str} | "
        f"{grad_str} | "
        f"{optimizer_str} | "
        f"{x_str} | "
        f"{other_str}"
    )



def model_parameter_cost(model, mode='display'):
    
    param_count = 0
    param_bytes = 0
    trainable_params = 0
    grad_count = 0
    grad_bytes = 0


    for name, param in model.named_parameters():
        num_params = param.numel()
        dtype = param.dtype
        bytes_per_param = param.element_size()
        total_bytes = num_params * bytes_per_param

        if param.grad is not None:
            grad_params = param.grad.numel()
            grad_dtype = param.grad.dtype
            grad_bytes_per_param = param.grad.element_size()
            grad_total_bytes = grad_params * grad_bytes_per_param
            grad_count += grad_params
            grad_bytes += grad_total_bytes


        param_count += num_params
        param_bytes += total_bytes
        status = 'TRAINABLE' if param.requires_grad else 'FROZEN'
        #logger.info(f"{name.ljust(40)} | {str(num_params).ljust(10)} | {str(round(total_bytes/1024**2, 2)).ljust(10)}MB | {dtype} | [{status}]")

        if param.requires_grad:
            trainable_params += num_params
    

    if mode == 'display':
        logger.info(
            f"[MODEL]  | "
            f"parameters = {param_count/1000000:.2f}M | "
            f"trainable = {trainable_params/1000000:.2f}M | "
            f"memory = {round(param_bytes/1024**2,2)}MB | "
            f"gradients = {grad_count/1000000:.2f}M | "
            f"gradients memory = {round(grad_bytes/1024**2,2)}MB "
        )
    elif mode == 'return':
        return param_bytes, grad_bytes


def optimizer_cost(optimizer, mode='display'):
    param_values = optimizer.state_dict()['state'].values()
    hyper_params = optimizer.state_dict()['param_groups'] # assume negigible

    total_memory = 0
    total_elements = 0
    for n, i in enumerate(param_values):

        param_memory = 0
        param_elements = 0
        
        for item,v in i.items():
            elements = v.numel()
            size = v.element_size()
            dtype = v.dtype
            memory = (size*elements)

            param_memory += memory
            param_elements += elements

        total_memory += param_memory
        total_elements += param_elements


    if mode == 'display':
        logger.info(
            f"[OPTIMIZER]  | "
            f"values = {total_elements/1000000:.2f}M | "
            f"memory = {round(total_memory,2)}MB "
            
            
        )

    elif mode == 'return':
        return total_memory

def get_model(frozen_backbone=False,grad_checkpointing=True):

    vit_model = timm.create_model(
    "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
)

    vit_model.load_state_dict(
        torch.load(
            "/scratch/prj/ccc_vit_finetuning/models/vit_large_patch16_224.dinov2.uni_mass100k/pytorch_model.bin"
            , map_location="cpu")
        , strict=True)

    if frozen_backbone:
        for param in vit_model.parameters():
            param.requires_grad = False

    if grad_checkpointing:
        vit_model.set_grad_checkpointing(True)
    else:
        vit_model.set_grad_checkpointing(False)
    
    task_model = nn.Linear(1024, 1)
    model = nn.Sequential(
        vit_model,
        task_model
    )
    return model

def run_trial(

    images = 100
    ,frozen_backbone = False
    ,grad_checkpointing = True
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.reset_peak_memory_stats()

    optimizer = None
    model = None
    x = None

    _log_gpu_mem("start",x,model,optimizer)

    x = torch.randn(images,3,224,224)
    x = x.to(device)
    _log_gpu_mem("data loaded",x,model,optimizer)


    model = get_model(
        frozen_backbone=frozen_backbone
        ,grad_checkpointing=grad_checkpointing)

    model.to(device)

    _log_gpu_mem("model loaded",x,model,optimizer)



    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    optimizer.zero_grad()
    _log_gpu_mem("batch start",x,model,optimizer)

    output = model(x)
    _log_gpu_mem("forward pass",x,model,optimizer)

    loss = F.mse_loss(output, torch.ones_like(output))
    _log_gpu_mem("loss calculation",x,model,optimizer)

    loss.backward()
    _log_gpu_mem("backward pass",x,model,optimizer)


    optimizer.step()
    _log_gpu_mem("optimizer step",x,model,optimizer)



    # second pass
    optimizer.zero_grad()
    _log_gpu_mem("batch 2 start",x,model,optimizer)

    output = model(x)
    _log_gpu_mem("forward pass 2",x,model,optimizer)

    loss = F.mse_loss(output, torch.ones_like(output))
    _log_gpu_mem("loss calculation 2",x,model,optimizer)

    loss.backward()
    _log_gpu_mem("backward pass 2",x,model,optimizer)

    optimizer.step()
    _log_gpu_mem("optimizer step 2",x,model,optimizer)



    # clean up GPU
    del x
    del model
    del optimizer
    del output
    del loss

    torch.cuda.empty_cache()

    logger.info(f"residual memory allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f}MB")


run = 0 

for frozen_backbone, grad_checkpointing in [(True, False), (False, True), (False, False)]:
    for i in [1,10,20,100,200,400,600,800,1000]:
        logger.info(f"#run {run}# IMAGES={i} # FROZEN={frozen_backbone} # CHECKPOINTING={grad_checkpointing}")
        run_trial(images=i,frozen_backbone=frozen_backbone,grad_checkpointing=grad_checkpointing)
        run += 1