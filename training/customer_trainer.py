# a built from scaratch training class for training the model

import logging
logger = logging.getLogger(__name__)
import torch
import numpy as np
import sys
from sksurv.metrics import concordance_index_censored

class CustomTrainer():

    def __init__(
        self
        ,lr_scheduler='constant'
        ,max_epochs=1
        ,device="cpu"):

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epoch_losses = []
        self.epoch_concept_losses = []
        self.epoch_task_losses = []
        self.epoch_val_losses = []
        self.epoch_val_c_index_history = []
        self.max_epochs = max_epochs
        self.epoch = 0
        self.lr_scheduler = lr_scheduler

    def _log_gpu_mem(self, tag=""):
        '''A useful helper to log VRAM usage'''

        if not torch.cuda.is_available():
            return

        device = torch.cuda.current_device()

        allocated = torch.cuda.memory_allocated(device) / 1024**2
        reserved = torch.cuda.memory_reserved(device) / 1024**2
        peak = torch.cuda.max_memory_allocated(device) / 1024**2

        logger.info(
            f"[{tag}] "
            f"allocated={allocated:.1f}MB | "
            f"reserved={reserved:.1f}MB | "
            f"peak={peak:.1f}MB"
        )


    def _move_to_device(self,obj, device):
        '''Move an object to a device (recursively if it is a tuple or list)'''
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        if isinstance(obj, (tuple, list)):
            return type(obj)(
                self._move_to_device(x, device) for x in obj
                )
        return obj

    def _prepare_batch(self, batch):
        return self._move_to_device(batch, self.device)

    def fit_epoch(self,train_dataloader):
        '''run an entire epoch for a given dataloader'''

        batch_idx = 0
        batch_total_losses = []
        batch_concept_losses = []
        batch_task_losses = []

       
        for batch in train_dataloader:
            
            # give GPU readouts every 10 batches
            if (batch_idx) % 10 == 0:
                self._log_gpu_mem(tag=f'epoch {self.epoch+1} | train | batch {batch_idx} start')

            
            batch = self._prepare_batch(batch)
            self.optim.zero_grad() # clear out the old gradients

            # https://docs.pytorch.org/docs/2.11/amp.html - exit before backward as need more preciion there
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                result = self.model.training_step(batch, batch_idx)
                loss = result.get("loss")

            # backprop    
            loss.backward()
            
            # update gradients
            self.optim.step()

            batch_total_losses.append(result['loss'].detach().cpu().numpy())
            batch_concept_losses.append(result.get('result').get('concept_loss').detach().cpu().numpy())
            batch_task_losses.append(result.get('result').get('task_loss').detach().cpu().numpy())

            batch_idx += 1


        epoch_total_loss = np.array(batch_total_losses).mean().round(3).item()
        epoch_concept_loss = np.array(batch_concept_losses).mean().round(3).item()
        epoch_task_loss = np.array(batch_task_losses).mean().round(3).item()

        self.epoch_losses.append(epoch_total_loss)
        self.epoch_concept_losses.append(epoch_concept_loss)
        self.epoch_task_losses.append(epoch_task_loss)

        logger.info(f'epoch {self.epoch+1} train total loss: {epoch_total_loss:.2f}')
        logger.info(f'epoch {self.epoch+1} train concept loss: {epoch_concept_loss:.2f}')
        logger.info(f'epoch {self.epoch+1} train task loss: {epoch_task_loss:.2f}')
        logger.info(f'epoch {self.epoch+1} train total loss history: {self.epoch_losses}')
        logger.info(f'epoch {self.epoch+1} train concept loss history: {self.epoch_concept_losses}')
        logger.info(f'epoch {self.epoch+1} train task loss history: {self.epoch_task_losses}')

    
    def _validate(self,val_dataloader):

        ''' 
        once per epoch (incl before, check validation set)
        returns epoch_val_loss and epoch_val_c_index
        '''
        batch_idx = 0
        batch_val_errors = []
        hazards_pred = []
        event_test = []
        survtime_test = []

        self.model.eval()  # set model to evaluation mode
        with torch.no_grad():

            for batch in val_dataloader:
                batch= self._prepare_batch(batch)

                with torch.no_grad():
                    result = self.model.validation_step(batch, batch_idx)

                # log GPU usage every 10
                if (batch_idx) % 10 == 0:
                    self._log_gpu_mem(tag=f'epoch {self.epoch+1} | val | batch {batch_idx} start')


                batch_val_errors.append(result['val_loss'].detach().cpu().numpy())

                # check C-index
                (event, survtime) = batch[1]
                event_test.append(event.detach().cpu().numpy())
                survtime_test.append(survtime.detach().cpu().numpy())

                batch_result = self.model.predict_step(batch, batch_idx)
                hazards = batch_result[1].detach().cpu().numpy()
                hazards_pred.append(hazards)

                batch_idx += 1

            # mean across all batches
            epoch_val_loss = np.array(batch_val_errors).mean().round(3).item()
    
            # calc c index
            hazards_pred = np.concatenate(hazards_pred, axis=0)
            event_test = np.concatenate(event_test, axis=0)
            survtime_test = np.concatenate(survtime_test, axis=0)
            y_test = np.array([(int(event_test[i]), int(survtime_test[i])) for i in range(len(event_test))])

            events = [bool(y[0]) for y in y_test]
            survtimes = [y[1] for y in y_test]

            try:
                c_index, _, _, _, _ = concordance_index_censored(
                    events, survtimes, hazards_pred.squeeze()
                    )
            except Exception as e:
                logger.error(f'Error calculating c-index: {e}')
                c_index = np.nan

            epoch_val_c_index = np.array(c_index).mean().round(3).item()

        
        self.epoch_val_losses.append(epoch_val_loss)
        self.epoch_val_c_index_history.append(epoch_val_c_index)
        logger.info(f'epoch {self.epoch+1} val loss (mean of batches): {epoch_val_loss:.2f}')
        logger.info(f'epoch {self.epoch+1} val loss history: {self.epoch_val_losses}')
        logger.info(f'epoch {self.epoch+1} val c-index (total sample): {epoch_val_c_index:.2f}')
        logger.info(f'epoch {self.epoch+1} val c-index history: {self.epoch_val_c_index_history}')

        
        self.model.train() # back to training mode


    def fit(self,model,train_dataloader,val_dataloader):
        '''fit the model provided using given datasets'''

       # reset memory for logging each time we fit a new model
        torch.cuda.reset_peak_memory_stats()

        self.model = model.to(self.device)

        # get the optimiser from the model and add training schedule
        raw_cfg = self.model.configure_optimizers()
        self.optim = raw_cfg.get("optimizer")

        if self.lr_scheduler == 'constant':
            self.lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                self.optim,
                factor=1.0
            )
        elif self.lr_scheduler == 'cosine':
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optim,
                T_max=self.max_epochs
            )

        logger.info(f'LR scheduler: {self.lr_scheduler}')
     

        # output model info before training - helpful for debugging
        logger.info("Parameters by layer:")
        total_params = 0
        trainable_params = 0
        for name, param in self.model.named_parameters():
            layer_params = param.numel()
            status = 'TRAINABLE' if param.requires_grad else 'FROZEN'
            logger.info(f"{name.ljust(40)} | {str(layer_params).ljust(10)} | [{status}]")
            total_params += layer_params
            if param.requires_grad:
                trainable_params += layer_params

        total_params_incl_unnamed = sum(p.numel() for p in model.parameters())
        assert total_params == total_params_incl_unnamed, "Total parameters mismatch: named params = {total_params}, model params = {total_params_incl_unnamed}"
        logger.info(f"Total parameters (incl unnamed): {total_params_incl_unnamed/1000000:.4f}M")
        logger.info(f"Total named parameters: {total_params/1000000:.4f}M")
        logger.info(f"Trainable parameters: {trainable_params/1000000:.4f}M")


        self.epoch = 0

        for e in range(self.max_epochs):

            self.epoch = e

            logger.info(f'======== starting epoch {self.epoch+1} ========')
            for i, group in enumerate(self.optim.param_groups):
                logger.info(f'epoch {self.epoch+1} LR at start of epoch: {group["name"]} = {group["lr"]:.6f}')
            
            # Start with validation (to match loss in train)
            logger.info(f'[VALIDATION STEP]')
            self.model.eval()  # set model to evaluation mode

            with torch.no_grad():   # no gradients for validation
                self._validate(val_dataloader)
            
            self.model.train() # back to training mode
            
            logger.info(f'[TRAINING STEP]')

            self.fit_epoch(train_dataloader)

            # update LR
            self.lr_scheduler.step()
            
            logger.info(f'epoch {self.epoch+1} complete')


    def predict(self,model,val_dataloader):
        ''' USe model to make predicitions, used when validating'''

        self.model = model.to(self.device)
  
        self.model.eval()  # set model to evaluation mode

        predictions = []

        batch_idx = 0

        with torch.no_grad():   # no gradients for validation
            for batch in val_dataloader:
                logger.info(f'  starting batch {batch_idx+1} of validation')
                batch= self._prepare_batch(batch)
                batch_pred = self.model.predict_step(batch, batch_idx)
                predictions.append(tuple(x.detach().cpu() for x in batch_pred))
                batch_idx += 1

        return predictions
