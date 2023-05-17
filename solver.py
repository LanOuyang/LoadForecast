import os 
import logging
import time

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.nn import MSELoss
from torch.utils.tensorboard import SummaryWriter
from tensorflow import summary
# from torchmetrics import MeanAbsolutePercentageError as MAPE

from visualizer import DynamicUpdate

logger = logging.getLogger(__name__)
log_dir = os.path.dirname(os.path.abspath(logging.root.handlers[0].baseFilename))
model_dir = os.path.join(log_dir, "models")
os.makedirs(model_dir, exist_ok=True)

class Solver(object):

    def __init__(self, 
                 model, 
                 optimizer,
                 optim_scheduler, 
                 loss_func=MSELoss(),
                 verbose=True, 
                 print_every=1,
                 **kwargs):
        
        self.model = model
        self.loss_func = loss_func

        self.opt = optimizer
        self.scheduler = optim_scheduler

        self.verbose = verbose
        self.print_every = print_every

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self._reset()

    def _reset(self):
        # Set up some variables for book-keeping
        self.best_model_stats = None
        self.best_params = None
        self.hidden_states = None

        self.train_loss_history = []
        self.val_loss_history = []

        self.train_batch_loss = []
        self.val_batch_loss = []

        self.current_patience = 0

    def _step(self, X, y, validation=False):
        loss = None

        y_pred, self.hidden_states = self.model.forward(X, self.hidden_states)
        
        loss = self.loss_func(y_pred,y)
        # Add the regularization
        # loss += sum(self.model.reg.values())

        # Perform gradient update (only in train mode)
        if not validation:
            # Compute gradients
            loss.backward()
            # Update weights
            self.opt.step()

        return y_pred, loss

    def train(self, 
              train_dataloader, 
              val_dataloader,
              summary_writer,
              output_scaler=None,
              epochs=100, 
              epoch_start = 0,
              print_freq = 500,
              patience=None, 
              validate_first_epoch=True):
        total_time = 0
        
        logger.info("Train starts")

        for t in range(epoch_start, epochs):
            self.model.train()

            train_epoch_loss = 0.0
            epoch_time = 0
            batch_start = time.time()

            for i, batch in enumerate(train_dataloader):
                X, y = batch

                X = X.float().to(self.device)
                y = y.float().to(self.device)

                self.opt.zero_grad()

                validate = t == 0
                y_pred, train_loss = self._step(X, y, validation=(validate and validate_first_epoch))

                train_loss = train_loss.item()
                if output_scaler:
                  train_loss = train_loss * output_scaler["scale"]**2
                self.train_batch_loss.append(train_loss)
                train_epoch_loss += train_loss

                with summary_writer.as_default():
                  summary.scalar('loss/train', train_loss, step=t*len(train_dataloader)+i)


                if (i+1) % print_freq == 0 or i == len(train_dataloader)-1:
                    logger.info("Iteration [{}]: elapsed time {:.2f}s, training loss {:.5f}".
                                format(i+1, time.time()-batch_start, np.mean(self.train_batch_loss)))
                    
                    batch_start = time.time()
                    self.train_batch_loss.clear()      

            train_epoch_loss /= len(train_dataloader)

            if self.scheduler:
                self.scheduler.step()

            val_epoch_loss = self.evaluate(val_dataloader, summary_writer, t, output_scaler)

            self.train_loss_history.append(train_epoch_loss)
            self.val_loss_history.append(val_epoch_loss)

            if self.verbose and t % self.print_every == 0:
                logger.info('(Epoch %d / %d) train loss: %.5f; val loss: %.5f' % (
                    t + 1, epochs, train_epoch_loss, val_epoch_loss))

            self.update_best_loss(val_epoch_loss, train_epoch_loss)

            save_path = os.path.join(model_dir, "model-"+str(t))
            print('Saving model to ', save_path)
            torch.save({
              'epoch': t,
              'state_dict': self.model.state_dict(),
              'optimizer_state_dict': self.opt.state_dict(),
              'scheduler_state_dict': self.scheduler.state_dict()}, save_path)
            
            if patience and self.current_patience >= patience:
                logger.info("Stopping early at epoch {}!".format(t+1))
                break

        self.model.parameters = self.best_params

    def evaluate(self, 
            dataloader, 
            summary_writer,
            current_epoch,
            output_scaler=None, 
            print_freq=500, 
            visualize=False):
        
        logger.info("Eval")
        start = time.time()
        total_loss = 0.0
        batch_loss = []
        if visualize:
            plot = DynamicUpdate()
        
        self.model.eval()
        self.hidden_states = None

        with torch.no_grad():

            for i, batch in enumerate(dataloader):
                X, y = batch

                X = X.float().to(self.device)
                y = y.float().to(self.device)

                y_pred, loss = self._step(X, y, validation=True)

                loss = loss.item()
                if output_scaler:
                  loss = loss * output_scaler["scale"]**2
                batch_loss.append(loss)
                total_loss += loss

                if visualize:
                    pred = (y_pred.squeeze().detach().cpu().numpy() * output_scaler["scale"] + output_scaler["mean"]) if output_scaler else y_pred.squeeze().detach().cpu().numpy()
                    target = (y.squeeze().detach().cpu().numpy() * output_scaler["scale"] + output_scaler["mean"]) if output_scaler else y.squeeze().detach().cpu().numpy()

                    ydata = np.stack((pred, target, np.abs(pred-target)), axis=1) 
                    xdata = np.arange((i)*len(pred), (i+1)*len(pred))
                    plot(xdata, ydata)

                with summary_writer.as_default():
                  summary.scalar('loss/eval', loss, step=current_epoch*len(dataloader)+i)

                if (i+1) % print_freq == 0 or i == len(dataloader)-1:

                    logger.info("Iteration [{}]: elapsed time {:.2f}(s), eval loss {:.5f}".
                                format(i+1, time.time()-start, total_loss/(i+1)))
                    
                    start = time.time()

            total_loss /= len(dataloader)
            logger.info("Eval total loss: {:.5f}".format(total_loss))

        return total_loss

    def update_best_loss(self, val_loss, train_loss):
        # Update the model and best loss if we see improvements.
        if not self.best_model_stats or val_loss < self.best_model_stats["val_loss"]:
            self.best_model_stats = {"val_loss": val_loss, "train_loss": train_loss}
            self.best_params = self.model.parameters
            self.current_patience = 0
        else:
            self.current_patience += 1







