
## This file builds the model class as outlined in the report

import logging
logger = logging.getLogger(__name__)

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys
from einops import rearrange
import timm

import torchvision.transforms as transforms
from peft import LoraConfig, get_peft_model


##############################################################  
from vit_cbm.models.losses import CoxLoss
# from vit_cbm.models.embedding_models import vit_model
##############################################################  
def vit_model(model_name):
    '''the ViT model used as embedding head in main model below'''


    # use UNI as our model
    if model_name == "uni":
        model = timm.create_model(
        "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
    )

        model.load_state_dict(
            torch.load(
                "/scratch/prj/ccc_vit_finetuning/models/vit_large_patch16_224.dinov2.uni_mass100k/pytorch_model.bin"
                , map_location="cpu")
            , strict=True)
    # flexilibty to add other options later
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    return model

class ABMIL(nn.Module):
    ''' implemented per https://arxiv.org/pdf/1802.04712 - used in model below'''
    def __init__(self,input_dim,hidden_dim, dropout=0.0):
        super().__init__()
        
        self.M = input_dim # input dimension
        self.L = hidden_dim # hidden dimension
      
        self.w_map = nn.Sequential(
            nn.Linear(self.L,1,bias=False)
            )

        self.v_map = nn.Sequential(
            nn.Linear(self.M,self.L,bias=False)
            , nn.Tanh()
            , nn.Dropout(dropout)
            )

    def forward(self, x):
        # x is [batch,num_tiles,M]
        v = self.v_map(x) # [batch,num_tiles,L]
        weights = self.w_map(v) # [batch,num_tiles,1]
        weights = F.softmax(weights, dim=1) # [batch,num_tiles,1]
        output = x*weights # [batch,num_tiles,M] where each x element is weighted by the weight
        output = output.sum(dim=1) # [batch,M]
        
        return output


class CBM_VIT(nn.Module):

    def __init__(self
        ,concept_states
        ,no_concepts=False
        ,vit_model_name='uni'
        ,optimizer='adam'
        ,learning_rate=0.001
        ,vit_learning_rate=None
        ,dropout=0.2
        ,weight_decay=0.00004
        ,fine_tune='none'
        ,grad_checkpointing=False
        ,pooling='mean'
        ,concept_heads='linear'
        ,task_head='linear'
        ,concept_loss_weight=1.0
        ):

        super().__init__()
        
        self.concept_states = concept_states
        self.no_concepts = no_concepts

        self.optimizer_name = optimizer
        self.learning_rate = learning_rate

        if vit_learning_rate is not None:
            self.vit_learning_rate = vit_learning_rate
        else:
            self.vit_learning_rate = self.learning_rate

        self.dropout = dropout
        self.weight_decay = weight_decay

        self.concept_loss_weight = concept_loss_weight

        self.vit_model_name = vit_model_name
        self.fine_tune = fine_tune
        self.pooling = pooling
        self.concept_head_type = concept_heads
        self.task_head_type = task_head
        self.grad_checkpointing = grad_checkpointing
        self.task_loss_fn = CoxLoss()


        # BUILD VIT ================================
        self.vit_model = vit_model(self.vit_model_name)

        self.embedding_dim = 1024 # uni default, needs updating if we add more
 
        # set parameters trainable based on finetuning mode
        if self.fine_tune == 'none':

            for param in self.vit_model.parameters():
                param.requires_grad = False


        elif self.fine_tune == 'full':

            for param in self.vit_model.parameters():
                param.requires_grad = True


        elif self.fine_tune == 'bottom-only':

            for param in self.vit_model.parameters():
                param.requires_grad = False

            # unfreeze last 2 blocks
            for block in self.vit_model.blocks[-1:]:
                for p in block.parameters():
                    p.requires_grad = True

            # unfreeze the final norm
            for p in self.vit_model.norm.parameters():
                p.requires_grad = True


        elif self.fine_tune == 'lora':

            config = LoraConfig(
            r=8,
            lora_alpha=8,
            target_modules=["qkv"],
            lora_dropout=0.1,
            )

            self.vit_model = get_peft_model(self.vit_model, config)

        else:
            raise ValueError(f"Invalid fine_tune: {self.fine_tune}")

        if self.grad_checkpointing:
            self.vit_model.set_grad_checkpointing(True)
        else:
            self.vit_model.set_grad_checkpointing(False)

        # BUILD MIL ================================
        if self.pooling == 'abmil':
            self.mil_model = ABMIL(self.embedding_dim, self.embedding_dim, self.dropout)
        elif self.pooling == 'mean':
            self.mil_model = None
        else:
            raise ValueError(f'Invalid pooling: {self.pooling}')


        # BUILD CONCEPT/TASK HEAD ================================
        if self.no_concepts:

            # no concepts mode 
            if self.task_head_type == 'linear':
                self.task_model = nn.Linear(self.embedding_dim, 1)
            elif self.task_head_type == 'mlp-h-2h-1':
                self.task_model = nn.Sequential(
                    nn.Linear(self.embedding_dim, 2*self.embedding_dim)
                    , nn.ReLU()
                    , nn.Linear(2*self.embedding_dim, 1)
                )
            else:
                raise ValueError(f'Invalid task_head: {self.task_head_type}')
    
        else:
            # concept mode
            k = np.sum(self.concept_states) # number of logits needed

            if self.concept_head_type == 'linear':

                # linear layer for each concept (k_classes is logit size for each)
                self.concept_heads = nn.ModuleList([
                    nn.Linear(self.embedding_dim, k_classes) for k_classes in self.concept_states
                ])

            elif self.concept_head_type == 'mlp-h-2h-1':
                self.concept_heads = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(self.embedding_dim, 2*self.embedding_dim)
                        , nn.ReLU()
                        , nn.Linear(2*self.embedding_dim, k_classes)
                    ) for k_classes in self.concept_states
                ])
            else:
                raise ValueError(f'Invalid task_head: {self.task_head}')

 
            if self.task_head_type == 'linear':

                self.task_model = nn.Linear(k, 1)

            elif self.task_head_type == 'mlp-k-2k-1':

                self.task_model = nn.Sequential(
                    nn.Linear(k, 2*k)
                    , nn.ReLU()
                    , nn.Linear(2*k, 1) )


    def _embed_vit(self,x):

        '''
        input: [batch,bag_element,3,H,W ]
        output: [batch,bag_element,embedding]
        '''

        patches = x #[batch,bag_element,3,H,W ]

        patches_flattened = rearrange(
            patches,
            'b bg c h w -> (b bg) c h w'
        )
        
        embeddings_tensor = self.vit_model(patches_flattened)

        embeddings_tensor = rearrange(
            embeddings_tensor
            ,'(b bg) dim -> b bg dim'
            ,b=patches.shape[0]
            ,bg=patches.shape[1]
        )
        assert embeddings_tensor.shape == (patches.shape[0], patches.shape[1], self.embedding_dim), f'embeddings_tensor.shape: {embeddings_tensor.shape} != (patches.shape[0], patches.shape[1], self.embedding_dim): {patches.shape[0], patches.shape[1], self.embedding_dim}'

        return embeddings_tensor

    def _mil_agg(self,embedding):

        '''
        aggregation of the embedding of the bag
        input: [batch,bag_element,embedding_dim]
        output: [batch,embedding_dim]
        '''

        if self.pooling == 'abmil':
            return self.mil_model(embedding) #[batch,embedding_dim]
        elif self.pooling == 'mean':
            return embedding.mean(dim=1) #[batch,embedding_dim]


    def _get_concept_loss(self, concept_logits, c):

        '''
        takes the concept logits and the concept labels and returns the loss
        input: [batch,sum(concept_states)]
        output: [batch,1]
        '''
        if self.no_concepts:
            return None
        else:
            idx = 0
            
            concept_losses = []

            for i, states in enumerate(self.concept_states):
    
                
                logits = concept_logits[:, idx:idx + states]
                labels = c[:, i].long() # needed for cross entropy loss

                concept_loss_fn = nn.CrossEntropyLoss() # weight by class?


                loss = concept_loss_fn(logits, labels)
                
                concept_losses.append(loss)
                idx += states # move to start of next concept
            
            # mean across all concepts
            total_concept_loss = torch.mean(torch.stack(concept_losses))

            return total_concept_loss

    def _get_task_loss(self, task_logits, y):

        '''
        takes the task logits and the task labels and returns the loss
        input: task_logits: [batch,1] y: [batch,(event, survtime)]
        output: [batch,1]
        '''

        event, survtime = y
        hazard_pred = task_logits.reshape(-1)

        
        task_loss = self.task_loss_fn(survtime, event, hazard_pred, device=task_logits.device)
        
        return task_loss


    def _forward(self, x, c=None, y=None, train=False):
      
        embedding = self._embed_vit(x) # raw tiles → (batch, bag, embed_dim)

        h_slide = self._mil_agg(embedding) #[batch,embedding_dim]

        if self.no_concepts:
            concept_logits = torch.zeros(h_slide.shape[0], np.sum(self.concept_states), device=h_slide.device) #[batch,sum(concept_states)]
            task_logits = self.task_model(h_slide) #[batch,1]
        else:
            concept_logits = [head(h_slide) for head in self.concept_heads]
            concept_logits = torch.cat(concept_logits, dim=-1) #[batch,sum(concept_states)]
            task_logits = self.task_model(concept_logits) #[batch,1]

        return tuple([concept_logits, task_logits])


    def forward(
        self,
        x,
        c=None,
        y=None,
        train=False,
    ):
        return self._forward(x, c=c, y=y, train=train)

    def predict_step(
        self,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):  
        x, y, c, wsi_id = batch[0], batch[1], batch[2], batch[3]
        return self._forward(x, c=c, y=y, train=False)


    # Main training/validation step
    def _run_step(
        self,
        batch,
        batch_idx,
        train=False,
    ):

        x, y, c, wsi_id = batch[0], batch[1], batch[2], batch[3]

        outputs = self._forward(x, c=c, y=y, train=train)

        concept_logits, task_logits = outputs

        if train and batch_idx == 0:
            logger.info(f'    task_logits from  {task_logits.min().item()} to {task_logits.max().item()}')
        
        # task loss
        task_loss = self._get_task_loss(task_logits, y)
        task_loss_scalar = task_loss.detach().cpu()
        
        # concept loss
        if self.no_concepts:

            # no concepts mode
            concept_loss_scalar = torch.tensor(0.0, device=task_logits.device)
            loss = task_loss

        else:
            # concept mode
            concept_loss = self._get_concept_loss(concept_logits, c)
            concept_loss_scalar = concept_loss.detach().cpu()

            loss = task_loss + (self.concept_loss_weight * concept_loss)


        result = {
            "concept_loss": concept_loss_scalar
            ,"task_loss": task_loss_scalar
            ,"loss": loss.detach().cpu()
        }

        return loss, result

    def training_step(self, batch, batch_no):
        loss, result = self._run_step(batch, batch_no, train=True)

        return {"loss": loss,"result": result}

    def validation_step(self, batch, batch_no):
        _, result = self._run_step(batch, batch_no, train=False)

        result = {
            "val_" + key: val
            for key, val in result.items()
        }
        
        return result

    def test_step(self, batch, batch_no):
        loss, result = self._run_step(batch, batch_no, train=False)
        # for name, val in result.items():
        #     self.log("test_" + name, val, prog_bar=True)
        return result['loss']

    def configure_optimizers(self):

        if self.fine_tune in ['full', 'bottom-only','lora']:

            # configure optimizer for vit and head separately

            vit_params = []
            head_params = []

            for name, param in self.named_parameters():
                if param.requires_grad:
                    if 'vit_model' in name:
                        vit_params.append(param)
                    else:
                        head_params.append(param)

            # TO DO: add no decay params for bias
            optimizer = torch.optim.AdamW([
                {"name": "vit", "params": vit_params, "lr": self.vit_learning_rate},
                {"name": "head", "params": head_params, "lr": self.learning_rate},
                ],
                weight_decay=self.weight_decay,
                )
        else:

            # configure optimizer for all parameters together
            optimizer = torch.optim.Adam([
                {"name": "all", "params": filter(lambda p: p.requires_grad, self.parameters()), "lr": self.learning_rate}
                ],
                weight_decay=self.weight_decay,
                )
          
        return {"optimizer": optimizer}
