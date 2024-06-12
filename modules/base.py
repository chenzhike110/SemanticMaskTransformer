import torch
import lightning.pytorch as pl


class BaseModel(pl.LightningModule):
    def __init__(self, cfg, **kargs):
        super().__init__()
        
        self.cfg = cfg

    def training_step(self, batch, batch_idx):
        raise NotImplementedError
    
    def validation_step(self, batch, batch_idx):
        raise NotImplementedError
    
    def configure_optimizers(self):
        from torch import optim as optim_module
        from torch.optim import lr_scheduler as lr_sched_module

        gen_params = [a[1] for a in self.named_parameters() if a[1].requires_grad]
        gen_optimizer_class = getattr(optim_module, self.cfg.train.optimizer.target)
        gen_optimizer = gen_optimizer_class(gen_params, **self.cfg.train.optimizer.params)

        lr_sched_class = getattr(lr_sched_module, self.cfg.train.lr_scheduler.target)
        gen_lr_scheduler = lr_sched_class(gen_optimizer, **self.cfg.train.lr_scheduler.params)

        schedulers = [
            {
                'scheduler': gen_lr_scheduler,
                'monitor': 'val_acc',
                'interval': 'epoch',
                'frequency': self.cfg.train.val_frequency
            },
        ]
        return [gen_optimizer], schedulers
    
    def optimizer_step(self, epoch_nb, batch_nb, optimizer, opt_closure):
        if self.trainer.global_step < self.cfg.train.warm_up_iter:
            lr_scale = min(1., float(self.trainer.global_step + 1) / float(self.cfg.train.warm_up_iter))
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.cfg.train.optimizer.params.lr

        optimizer.step(closure=opt_closure)

    def on_save_checkpoint(self, checkpoint: torch.Dict[str, torch.Any]) -> None:
        checkpoint['cfg'] = self.cfg

        


        


        
        

