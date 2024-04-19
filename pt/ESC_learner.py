# Library
import copy
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from .net import Net
from typing import Union
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score

from torch.utils.tensorboard import SummaryWriter
from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReservedHeaderKey, Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.learner_spec import Learner
from nvflare.app_common.app_constant import AppConstants, ModelName, ValidateType
from nvflare.app_opt.pt.fedproxloss import PTFedProxLoss
from nvflare.fuel.utils.deprecated import deprecated
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.utils.fl_model_utils import FLModelUtils
from nvflare.apis.fl_constant import FLMetaKey


## Train Data
class trainData(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)


## Test Data    
class testData(Dataset):
    
    def __init__(self, X_data):
        self.X_data = X_data
        
    def __getitem__(self, index):
        return self.X_data[index]
        
    def __len__ (self):
        return len(self.X_data)

# Class EyeStateClassification learner:
class EYESTATELearner (Learner):
    def __init__(
        self,
        train_idx_root: str = "./dataset",
        subject_data: str = "S001",
        aggregation_epochs: int = 1,
        lr: float = 1e-2,
        fedproxloss_mu: float = 0.0,
        central: bool = False,
        analytic_sender_id: str = "analytic_sender",
        batch_size: int = 64,
        num_workers: int = 0,
    ):
        """Simple EyeStateClassification Trainer.

        Args:
            train_idx_root: directory with site training indices for CIFAR-10 data.
            aggregation_epochs: the number of training epochs for a round. Defaults to 1.
            lr: local learning rate. Float number. Defaults to 1e-2.
            fedproxloss_mu: weight for FedProx loss. Float number. Defaults to 0.0 (no FedProx).
            central: Bool. Whether to simulate central training. Default False.
            analytic_sender_id: id of `AnalyticsSender` if configured as a client component.
                If configured, TensorBoard events will be fired. Defaults to "analytic_sender".
            batch_size: batch size for training and validation.
            num_workers: number of workers for data loaders.

        Returns:
            a Shareable with the updated local model after running `execute()`
            or the best local model depending on the specified task.
        """
        super().__init__()
        # trainer init happens at the very beginning, only the basic info regarding the trainer is set here
        # the actual run has not started at this point
        self.train_idx_root = train_idx_root
        self.subject_data = subject_data
        self.aggregation_epochs = aggregation_epochs
        self.lr = lr
        self.fedproxloss_mu = fedproxloss_mu
        self.best_acc = 0.0
        self.central = central
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.writer = None
        self.analytic_sender_id = analytic_sender_id

        # Epoch counter
        self.epoch_of_start_time = 0
        self.epoch_global = 0
        self.epoch_learn = 0

        # following will be created in initialize() or later
        self.app_root = None
        self.client_id = None
        self.local_model_file = None
        self.best_local_model_file = None
        self.writer = None
        self.device = None
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.criterion_prox = None
        self.transform_train = None
        self.transform_valid = None
        self.train_dataset = None
        self.valid_dataset = None
        #self.valid_y = None
        self.test_dataset = None
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None

    def initialize(self, parts: dict, fl_ctx: FLContext):
        """
        Note: this code assumes a FL simulation setting
        Datasets will be initialized in train() and validate() when calling self._create_datasets()
        as we need to make sure that the server has already downloaded and split the data.
        """
        self.app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)
        fl_args = fl_ctx.get_prop(FLContextKey.ARGS)
        self.client_id = fl_ctx.get_identity_name()

        # when the run starts, this is where the actual settings get initialized for trainer
        self.log_info(fl_ctx,
            f"Client {self.client_id} initialized at \n {self.app_root} \n with args: {fl_args}",
        )

        self.local_model_file = os.path.join(self.app_root, "local_model.pt")
        self.best_local_model_file = os.path.join(self.app_root, "best_local_model.pt")

        # Select local TensorBoard writer or event-based writer for streaming
        self.writer = parts.get(self.analytic_sender_id)  # user configured config_fed_client.json for streaming
        if not self.writer:  # use local TensorBoard writer only
            self.writer = SummaryWriter(self.app_root)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = Net().to(self.device)
        # Can be modified
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = torch.nn.BCEWithLogitsLoss()

        if self.fedproxloss_mu > 0:
            self.log_info(fl_ctx,f"using FedProx loss with mu {self.fedproxloss_mu}")
            self.criterion_prox = PTFedProxLoss(mu=self.fedproxloss_mu)
        
        sub_train = self.train_idx_root + self.subject_data + '_train.csv'
        sub_valid = self.train_idx_root + self.subject_data + '_valid.csv'
        sub_test = self.train_idx_root + self.subject_data + '_test.csv'

        data_train = pd.read_csv(sub_train)
        self.X_train = data_train.iloc[:,0:64]
        self.y_train = data_train.iloc[:,64]
        data_valid = pd.read_csv(sub_valid)
        self.X_valid = data_valid.iloc[:,0:64]
        self.y_valid = data_valid.iloc[:,64]
        data_test = pd.read_csv(sub_test)
        self.X_test = data_test.iloc[:,0:64]
        self.y_test = data_test.iloc[:,64]
        random_state_set = 1000
        # Data preparation
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_valid = scaler.fit_transform(self.X_valid)
        self.X_test = scaler.fit_transform(self.X_test)
        self.train_dataset = trainData(torch.FloatTensor(self.X_train), torch.FloatTensor(self.y_train.values))
        self.valid_dataset = trainData(torch.FloatTensor(self.X_valid), torch.FloatTensor(self.y_valid.values))
        self.test_dataset = testData(torch.FloatTensor(self.X_test))
        #self.valid_y = y_valid

        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True ,num_workers=self.num_workers)
        self.valid_loader = DataLoader(dataset=self.valid_dataset, batch_size=1, shuffle=False ,num_workers=self.num_workers)
        self.test_loader = DataLoader(dataset=self.test_dataset, batch_size=1)
        self.epoch_learn = 50

    def local_train(self, fl_ctx, train_loader, model_global, abort_signal:Signal, val_freq: int = 0):
        #for epoch in range(self.aggregation_epochs):
        for epoch in range(self.epoch_learn):
            self.model.train()
            epoch_len = len(train_loader)
            self.epoch_global = self.epoch_of_start_time + epoch
            self.log_info(fl_ctx,f"Local epoch {self.client_id}: {epoch + 1}/{self.epoch_learn} (lr={self.lr})")
            avg_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader):
                #self.log_info(fl_ctx,f"input type : {type(inputs)}")
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # zero the parameter gradients
                self.optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels.unsqueeze(1))

                # FedProx loss term
                if self.fedproxloss_mu > 0:
                    fed_prox_loss = self.criterion_prox(self.model, model_global)
                    loss += fed_prox_loss

                loss.backward()
                self.optimizer.step()
                current_step = epoch_len * self.epoch_global + i
                avg_loss += loss.item()
            self.writer.add_scalar("train_loss", avg_loss / len(train_loader), current_step)
            self.log_info(fl_ctx,f"Local epoch {self.client_id}:{epoch + 1}/{self.epoch_learn} loss {avg_loss/ len(train_loader)}")
            acc = self.local_valid(self.valid_loader, abort_signal, tb_id="val_acc_local_model", fl_ctx=fl_ctx,during_epoch = True)
            self.log_info(fl_ctx,f"Local epoch {self.client_id}:{epoch + 1}/{self.epoch_learn} acc {acc}")
            if val_freq > 0 and epoch % val_freq == 0:
                #acc = self.local_valid(self.valid_loader, tb_id="val_acc_local_model")
                acc = self.local_valid(self.valid_loader, abort_signal, tb_id="val_acc_local_model", fl_ctx=fl_ctx)
                if acc > self.best_acc:
                    self.best_acc = acc
                    self.save_model(is_best=True)

    def save_model(self, is_best=False):
        # save model
        model_weights = self.model.state_dict()
        save_dict = {"model_weights": model_weights, "epoch": self.epoch_global}
        if is_best:
            save_dict.update({"best_acc": self.best_acc})
            torch.save(save_dict, self.best_local_model_file)
        else:
            torch.save(save_dict, self.local_model_file)

    def train(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:

        # get round information
        current_round = shareable.get_header(AppConstants.CURRENT_ROUND)
        total_rounds = shareable.get_header(AppConstants.NUM_ROUNDS)
        self.log_info(fl_ctx,f"Current/Total Round: {current_round + 1}/{total_rounds}")
        self.log_info(fl_ctx,f"Client identity: {fl_ctx.get_identity_name()}")

        # update local model weights with received weights
        dxo = from_shareable(shareable)
        global_weights = dxo.data

        # Before loading weights, tensors might need to be reshaped to support HE for secure aggregation.
        local_var_dict = self.model.state_dict()
        model_keys = global_weights.keys()
        for var_name in local_var_dict:
            if var_name in model_keys:
                weights = global_weights[var_name]
                try:
                    # reshape global weights to compute difference later on
                    global_weights[var_name] = np.reshape(weights, local_var_dict[var_name].shape)
                    # update the local dict
                    local_var_dict[var_name] = torch.as_tensor(global_weights[var_name])
                except BaseException as e:
                    raise ValueError(f"Convert weight from {var_name} failed") from e
        self.model.load_state_dict(local_var_dict)

        # local steps
        epoch_len = len(self.train_loader)
        self.log_info(fl_ctx,f"Local steps per epoch: {epoch_len}")

        # make a copy of model_global as reference for potential FedProx loss or SCAFFOLD
        model_global = copy.deepcopy(self.model)
        for param in model_global.parameters():
            param.requires_grad = False

        # local train
        self.local_train(
            fl_ctx=fl_ctx,
            train_loader=self.train_loader,
            model_global=model_global,
            abort_signal=abort_signal,
            val_freq=1 if self.central else 0,
        )
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)
        self.epoch_of_start_time += self.aggregation_epochs

        # perform valid after local train
        acc = self.local_valid(self.valid_loader, abort_signal, tb_id="val_acc_local_model", fl_ctx=fl_ctx)
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)
        self.log_info(fl_ctx, f"val_acc_local_model: {acc:.4f}")

        # save model
        self.save_model(is_best=False)
        if acc > self.best_acc:
            self.best_acc = acc
            self.save_model(is_best=True)

        # compute delta model, global model has the primary key set
        local_weights = self.model.state_dict()
        model_diff = {}
        for name in global_weights:
            if name not in local_weights:
                continuev
            model_diff[name] = np.subtract(local_weights[name].cpu().numpy(), global_weights[name], dtype=np.float32)
            if np.any(np.isnan(model_diff[name])):
                self.stop_task(f"{name} weights became NaN...")
                return ReturnCode.EXECUTION_EXCEPTION

        # build the shareable
        dxo = DXO(data_kind=DataKind.WEIGHT_DIFF, data=model_diff)
        dxo.set_meta_prop(MetaKey.NUM_STEPS_CURRENT_ROUND, epoch_len)

        self.log_info(fl_ctx, "Local epochs finished. Returning shareable")
        return dxo.to_shareable()
    
    def get_model_for_validation(self, model_name: str, fl_ctx: FLContext) -> Shareable:
        # Retrieve the best local model saved during training.
        if model_name == ModelName.BEST_MODEL:
            try:
                # load model to cpu as server might or might not have a GPU
                model_data = torch.load(self.best_local_model_file, map_location="cpu")
            except Exception as e:
                raise ValueError("Unable to load best model") from e

            # Create FLModel from model data.
            if model_data:
                # convert weights to numpy to support FOBS
                model_weights = model_data["model_weights"]
                for k, v in model_weights.items():
                    model_weights[k] = v.numpy()
                return FLModel(params_type=ParamsType.FULL, params=model_weights)
            else:
                # Set return code.
                self.error(f"best local model not found at {self.best_local_model_file}.")
                return ReturnCode.EXECUTION_RESULT_ERROR
        else:
            raise ValueError(f"Unknown model_type: {model_name}")  # Raised errors are caught in LearnerExecutor class.

    def local_valid(self, valid_loader, abort_signal: Signal, tb_id=None, fl_ctx=None, during_epoch=False):
        y_pred_list = []
        self.model.eval()
        with torch.no_grad():
            correct, total = 0, 0
            #self.log_info(fl_ctx,f"valid loader {valid_loader}")
            for _i, (inputs, labels) in enumerate(valid_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                outputs = torch.sigmoid(outputs)
                pred_label = torch.round(outputs)
                #_, pred_label = torch.max(outputs.data, 1)
                y_pred_list.append(pred_label.cpu().numpy())

                total += inputs.data.size()[0]
                correct += (pred_label == labels.data).sum().item()
            metric = correct / float(total)
            if tb_id:
                self.writer.add_scalar(tb_id, metric, self.epoch_global)
        if during_epoch == True:
            y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
            tn, fp, fn, tp = confusion_matrix(self.y_valid, y_pred_list).ravel()
            self.log_info(fl_ctx,f"valid_Accuracy {accuracy_score(self.y_valid, y_pred_list)}")
            self.log_info(fl_ctx,f"valid_Recall {tp/(tp+fn)}")
            self.log_info(fl_ctx,f"valid_Precision {tp/(tp+fp)}")
            self.log_info(fl_ctx,f"valid_F1 {(2*(tp/(tp+fp))*(tp/(tp+fn)))/((tp/(tp+fp))+(tp/(tp+fn)))}")
        return metric

    def validate(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        # Check abort signal
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        # get validation information
        self.log_info(fl_ctx, f"Client identity: {fl_ctx.get_identity_name()}")
        model_owner = shareable.get(ReservedHeaderKey.HEADERS).get(AppConstants.MODEL_OWNER)
        if model_owner:
            self.log_info(fl_ctx, f"Evaluating model from {model_owner} on {fl_ctx.get_identity_name()}")
        else:
            model_owner = "global_model"  # evaluating global model during training

        # update local model weights with received weights
        dxo = from_shareable(shareable)
        global_weights = dxo.data

        # Before loading weights, tensors might need to be reshaped to support HE for secure aggregation.
        local_var_dict = self.model.state_dict()
        model_keys = global_weights.keys()
        n_loaded = 0

        for var_name in local_var_dict:
            if var_name in model_keys:
                weights = torch.as_tensor(global_weights[var_name], device=self.device)
                try:
                    # update the local dict
                    local_var_dict[var_name] = torch.as_tensor(torch.reshape(weights, local_var_dict[var_name].shape))
                    n_loaded += 1
                except BaseException as e:
                    raise ValueError(f"Convert weight from {var_name} failed") from e
        self.model.load_state_dict(local_var_dict)
        if n_loaded == 0:
            raise ValueError(f"No weights loaded for validation! Received weight dict is {global_weights}")

        validate_type = shareable.get_header(AppConstants.VALIDATE_TYPE)
        if validate_type == ValidateType.BEFORE_TRAIN_VALIDATE:
            # perform valid before local train
            global_acc = self.local_valid(self.valid_loader, abort_signal, tb_id="val_acc_global_model", fl_ctx=fl_ctx)
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            self.log_info(fl_ctx, f"val_acc_global_model ({model_owner}): {global_acc}")

            return DXO(data_kind=DataKind.METRICS, data={MetaKey.INITIAL_METRICS: global_acc}, meta={}).to_shareable()

        elif validate_type == ValidateType.MODEL_VALIDATE:
            # perform valid
            train_acc = self.local_valid(self.train_loader, abort_signal)
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            self.log_info(fl_ctx, f"training acc ({model_owner}): {train_acc}")

            val_acc = self.local_valid(self.valid_loader, abort_signal)
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            self.log_info(fl_ctx, f"validation acc ({model_owner}): {val_acc}")

            self.log_info(fl_ctx, "Evaluation finished. Returning shareable")

            val_results = {"train_accuracy": train_acc, "val_accuracy": val_acc}

            metric_dxo = DXO(data_kind=DataKind.METRICS, data=val_results)
            return metric_dxo.to_shareable()

        else:
            return make_reply(ReturnCode.VALIDATE_TYPE_UNKNOWN)
