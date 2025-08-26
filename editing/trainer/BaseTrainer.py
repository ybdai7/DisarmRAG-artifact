import json
import logging
import os
import shutil
import tempfile
import time
from copy import deepcopy
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import copy
from .losses import kl_loc_loss
from .utils import *
from omegaconf import OmegaConf
from .models import *
from torch.utils.data import Dataset, DataLoader
from ..util.alg_train_dict import ALG_TRAIN_DICT
import importlib
from .utils import (
    EarlyStopper,
    RunningStatAverager,
    _logits,
    formatted_timestamp,
    safe_backward,
    time_delta_seconds,
)

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)

LOG = logging.getLogger(__name__)


class BaseTrainer:
    def __init__(self, config, train_set: Dataset, val_set: Dataset, save_name: str=None, pre_edited_model=None):
        LOG.info(f'Config: {config}')
        model_ = get_model(config, pre_edited_model)
        if 'qwen2' in config.model_name.lower():
            model_.bfloat16()
        self.alg_module = ALG_TRAIN_DICT[config.alg.upper()]
        LOG.info(f"Loading class {config.alg.upper()} from module {self.alg_module}")
        self.model = self.alg_module(model_, config, lambda: copy.deepcopy(model_))

        self.config = config

        self.save_name = save_name

        if config.train_base:
            self.original_model = self.model.model_constructor()
            self.original_model.load_state_dict(self.model.model.state_dict())
            self.original_model.to(self.config.device)
        else:
            self.original_model = self.model.model
        
        if self.config.model_parallel:
            self.config.device = self.model.model.device
        if not self.config.model_parallel and hasattr(self.config, 'device'):
            self.model.to(self.config.device)

        self.train_set = train_set
        self.val_set = val_set

        if 'minigpt4' in self.config.model_name.lower() or 'blip2' in self.config.model_name.lower():
            collate_fn = train_set.collate_fn
        elif 't5' in self.config.model_class.lower():
            collate_fn = train_set.collate_fn
        elif 'gpt' in self.config.model_class.lower():
            collate_fn = train_set.collate_gpt_fn
        elif 'llama' in self.config.model_class.lower():
            collate_fn = train_set.collate_gpt_fn
        elif 'qwen' in self.config.model_name.lower():
            collate_fn = train_set.collate_gpt_fn
        elif 'mistral' in self.config.model_name.lower():
            collate_fn = train_set.collate_gpt_fn
        elif 'contriever' in self.config.model_name.lower():
            collate_fn = train_set.collate_fn
        elif "ance" in self.config.model_name.lower():
            collate_fn = train_set.collate_fn
        elif "simcse" in self.config.model_name.lower():
            collate_fn = train_set.collate_fn
        else:
            raise NotImplementedError(f'Model {self.config.model_class} not supported yet.')

        self.train_loader = DataLoader(train_set, batch_size=self.config.batch_size,
                                       shuffle=True, collate_fn=collate_fn)
        self.val_loader = DataLoader(val_set, batch_size=self.config.val_batch_size,
                                       shuffle=False, collate_fn=collate_fn)

        if self.config.eval_only:
            # Eval once and quit
            self.config.max_iters = 0

        if not self.config.eval_only and self.config.alg!='MALMEN':
            self.OptimizerClass = getattr(torch.optim, config.opt)
            LOG.info(f"Building optimizer {self.OptimizerClass} with lr {config.lr}")
            self.opt = self.OptimizerClass(self.model.outer_parameters(), lr=config.lr)

        if config.archive is not None:
            archive, config.archive = load_archive(str(config.archive))
            self.model.load_state_dict(archive["model"])
            del archive["model"]
            if not self.config.eval_only:
                if self.config.alg=='MALMEN':
                    self.model.opt.load_state_dict(archive["opt"])
                else:
                    self.opt.load_state_dict(archive["opt"])
            del archive["opt"]

            self.archive = (
                archive  # Save for later to load e.g. lr_opt params if they exist
            )
        else:
            self.archive = None

        # # outfiles
        # with open(os.getcwd() + "/config.json", "w") as f:
        #     json.dump(OmegaConf.to_container(config), f)

        # already ./results/MEND/TRAINING/contriever/models/MEND
        model_dir = os.path.join(config.results_dir, "models", config.alg)
        if not (self.config.debug and not self.config.save) and not os.path.exists(model_dir):
            os.makedirs(model_dir)
        safe_model_name = self.config.model_name.split("/")[-1]  # Make sure no slashes

        self.save_path = f"{model_dir}/{safe_model_name}"

        self.save_path_using = f"{model_dir}/{self.save_name}"

        self.start_time = formatted_timestamp()

    def save_state(self, stats):
        if (self.config.debug and not self.config.save) or self.config.eval_only:
            return

        obj = {
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict() if self.config.alg!='MALMEN' else self.model.opt.state_dict(),
            "lr_opt": self.lr_opt.state_dict() if self.lr_opt is not None else None,
            "val_stats": stats,
            "start_time": self.start_time,
            "elapsed_time": time_delta_seconds(self.start_time),
            "step": self.global_iter,
        }
        LOG.info(f"Saving model to {self.save_path_using}")

        if os.path.exists(self.save_path_using):
            bk_path = f"{self.save_path_using}.bk"
            LOG.info(f"Moving old archive to {bk_path}")
            os.rename(self.save_path_using, bk_path)

        torch.save(obj, self.save_path_using)
        LOG.info("Write complete.")
    
    def norm_distance_loss(self, edited_model, original_param):
        norm_distance_loss = 0
        inner_params_in_mend = ["model."+layer_name for layer_name in self.config.inner_params]
        for n, p in edited_model.named_parameters():
            if n in inner_params_in_mend:
                name_win_model = n[6:]
                norm_distance_loss += (p.data - original_param[name_win_model].data).norm()
        return norm_distance_loss
    
    def outer_param_norm_loss(self):
        outer_param_norm_loss = 0
        for p in self.model.outer_parameters()[:-1]:
            outer_param_norm_loss += p.data.norm()
        return outer_param_norm_loss 

    def inbatch_ct_loss(self, anchor, positive, negative, temperature=0.07):
        # Normalize
        anchor = F.normalize(anchor, dim=1)
        positive = F.normalize(positive, dim=1)
        negative = F.normalize(negative, dim=1)
        
        pos_sim = torch.sum(anchor * positive, dim=1) / temperature
        neg_sim = torch.sum(anchor * negative, dim=1) / temperature

        logits = torch.stack([pos_sim, neg_sim], dim=1)
        labels = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)
        loss = F.cross_entropy(logits, labels)
        return loss
    
    def crossbatch_ct_loss(self, anchor, positive, negative, temperature=0.07):
        #anchor:   [B, D]
        anchor = F.normalize(anchor, dim=1)
        positive = F.normalize(positive, dim=1)
        negative = F.normalize(negative, dim=1)

        B = anchor.size(0)

        pos_sim = 1.3 * torch.matmul(anchor, positive.T) / temperature  # 正样本 logits
        neg_sim = torch.matmul(anchor, negative.T) / temperature  # 负样本 logits

        logits = torch.cat([pos_sim, neg_sim], dim=1)  # shape: [B, 2B]
        labels = torch.arange(B, device=anchor.device)
        loss = F.cross_entropy(logits, labels)
        return loss

    def crossbatch_ct_loss_pos1_allposneg(self, anchor, positive, negative, temperature: float = 0.07):
        """
        对每个 anchor:
        - 正样本: positive[i]
        - 负样本: 所有其它正样本 positive[j], j != i  +  全部 negative[k]
        最终分类大小: (1 + (B-1) + B) = 2B
        """
        # L2 归一化
        anchor = F.normalize(anchor, dim=1)
        positive = F.normalize(positive, dim=1)
        negative = F.normalize(negative, dim=1)

        device = anchor.device
        B = anchor.size(0)
        assert positive.size(0) == B and negative.size(0) == B, "anchor/positive/negative batch size must match"

        # 与所有 positive 的相似度: [B, B]
        pos_all = torch.matmul(anchor, positive.T) / temperature

        # 取对角线上作为唯一正样本: [B]
        pos_diag = pos_all.diag()  # positive[i] 与 anchor[i]

        # 其余 positive 作为负样本: [B, B-1]
        # 通过 mask 去掉对角线（本行自身正样本）
        mask_offdiag = ~torch.eye(B, dtype=torch.bool, device=device)
        pos_off_as_neg = pos_all[mask_offdiag].view(B, B - 1)

        # 与所有 negative 的相似度: [B, B]
        neg_all = torch.matmul(anchor, negative.T) / temperature

        # 组装 logits: [pos(i,i) | pos(i, j!=i) | neg(i, :)] -> [B, 1 + (B-1) + B] = [B, 2B]
        logits = torch.cat([
            pos_diag.unsqueeze(1),   # [B, 1]
            pos_off_as_neg,          # [B, B-1]
            neg_all                  # [B, B]
        ], dim=1)

        # 标签：正类在第 0 列
        labels = torch.zeros(B, dtype=torch.long, device=device)

        loss = F.cross_entropy(logits, labels)
        return loss

    def echo(self, train_step, info_dict, pretty=True):
        if not self.config.silent:
            sep = "\n" if pretty else "; "

            def key_format(k):
                return k.ljust(20) if pretty else k

            LOG.info(f"Step {train_step}:")
            LOG.info(
                "\n"+sep.join([f"{key_format(k)}: {v: 0.5f}" for k, v in info_dict.items()])
            )

    def evaluate_edit(self, return_edited_model: bool = False): 
        d = torch.load(self.save_path_using, map_location='cpu')
        self.model.load_state_dict(
            {k.replace("gtn.", "mend."): v for k, v in d["model"].items()}
        )
        global_iter = 0
        if return_edited_model:
            val_info, edited_contriever = self.validate(steps=self.config.val_steps, return_edited_model=return_edited_model)
        else:
            val_info = self.validate(steps=self.config.val_steps)
        self.echo(global_iter, val_info)

        if return_edited_model:
            return edited_contriever
        else:
            return True
    
    def cooptimize_run(self):
        averager = RunningStatAverager("train")
        self.global_iter = 0

        assert self.config.max_epochs is not None or self.config.max_iters is not None

        if self.config.max_epochs is not None:
            if self.config.max_iters is not None:
                self.config.max_iters = min(self.config.max_iters, self.config.max_epochs * len(self.train_set))
            else:
                self.config.max_iters = self.config.max_epochs * len(self.train_set)
            LOG.info(f'MAX EPOCH: {self.config.max_epochs}, set max iters to {self.config.max_iters}')

        self.epoches = round(float(self.config.max_iters) / (len(self.train_set) / self.config.batch_size))
        if self.epoches < 1:
            self.epoches = 1
        self.global_iter = 0
        should_stop = False
        current_train_loss = 0

        for epoch in range(self.epoches):
            if should_stop:
                break
            for i, batch in enumerate(self.train_loader):
                if self.global_iter >= self.config.max_iters:
                    should_stop = True
                    break

                if not self.config.eval_only:
                    train_info = self.train_step(batch)
                    averager.add(train_info)

                    if self.global_iter % self.config.log_interval == 0:
                        avg_info = averager.average()
                        averager.reset()
                        current_train_loss = avg_info['loss/edit_train']

                self.global_iter += 1
        
        return current_train_loss

    def run(self):
        averager = RunningStatAverager("train")
        stopper = EarlyStopper(
            self.config.early_stop_patience, self.config.early_stop_key
        )
        self.global_iter = 0

        assert self.config.max_epochs is not None or self.config.max_iters is not None
        if self.config.max_epochs is not None:
            if self.config.max_iters is not None:
                self.config.max_iters = min(self.config.max_iters, self.config.max_epochs * len(self.train_set))
            else:
                self.config.max_iters = self.config.max_epochs * len(self.train_set)
            if self.config.alg == 'MALMEN':
                self.config.max_iters = math.ceil(self.config.max_iters / self.config.batch_size)
            LOG.info(f'MAX EPOCH: {self.config.max_epochs}, set max iters to {self.config.max_iters}')
        if self.config.alg == 'MALMEN':
            n_edits_step = math.ceil(self.config.n_edits / self.config.batch_size)
            if self.config.log_interval % n_edits_step:
                self.config.log_interval = (self.config.log_interval // n_edits_step) * n_edits_step if self.config.log_interval >= n_edits_step else n_edits_step
            if self.config.val_interval % n_edits_step:
                self.config.val_interval = (self.config.val_interval // n_edits_step) * n_edits_step if self.config.val_interval >= n_edits_step else n_edits_step
        self.epoches = round(float(self.config.max_iters) / (len(self.train_set) / self.config.batch_size))
        if self.epoches < 1:
            self.epoches = 1
        self.global_iter = 0
        should_stop = False
        n_edits_batch = []
        for epoch in range(self.epoches):
            if should_stop:
                break
            for i, batch in enumerate(self.train_loader):
                # here should exist a check for the contriever performance on
                # usual benchmarks

                if self.global_iter >= self.config.max_iters:
                    should_stop = True
                    break
                if not self.config.eval_only:
                    if self.config.alg == 'MALMEN':  
                        n_edits_batch.append(batch)
                        if len(n_edits_batch) == math.ceil(self.config.n_edits / self.config.batch_size):
                            train_info = self.model.train(n_edits_batch)
                            averager.add(train_info)
                            n_edits_batch = []
                    else:
                        train_info = self.train_step(batch)
                        averager.add(train_info)

                    if self.global_iter % self.config.log_interval == 0:
                        avg_info = averager.average()
                        averager.reset()
                        self.echo(self.global_iter, avg_info)
                if self.global_iter % self.config.val_interval == 0:
                    if self.config.alg == 'MALMEN':
                        val_info = self.model.valid(config=self.config, loader=self.val_loader, val_set=self.val_set, steps=self.config.val_steps)
                    else:
                        val_info = self.validate(steps=self.config.val_steps)
                    self.echo(self.global_iter, val_info)
                    if True:
                        self.save_state(val_info)  # New best
                    if stopper.should_stop():
                        LOG.info(
                            f"No decrease in {self.config.early_stop_key} for {self.config.early_stop_patience} steps"
                        )
                        should_stop = True
                        break

                self.global_iter += 1

        if not self.config.eval_only:
            LOG.info(f"Training complete after {self.global_iter+1} steps.")

        if not self.config.final_eval:
            return

        if not self.config.eval_only:
            if (not self.config.debug) or self.config.save:
                if self.config.model_parallel:
                    archive = torch.load(self.save_path_using)
                    LOG.info(
                        f"Loading best model from step {archive['step']}, elapsed time {archive['elapsed_time']}"
                    )
                    self.model.load_state_dict(archive["model"])
                else:
                    archive = torch.load(self.save_path_using, map_location="cpu")
                    LOG.info(
                        f"Loading best model from step {archive['step']}, elapsed time {archive['elapsed_time']}"
                    )
                    self.model.to("cpu")
                    self.model.load_state_dict(archive["model"])
                    self.model.to(self.config.device)

        val_steps = self.config.val_steps if self.config.debug else None
        if self.config.alg == 'MALMEN':
            val_info = self.model.valid(log=True, steps=val_steps, config=self.config, loader=self.val_loader, val_set=self.val_set)
        else:
            val_info = self.validate(log=True, steps=val_steps)
        self.echo(self.global_iter, val_info, pretty=True)

        if self.config.results_dir is not None:
            results_path = f"{self.config.results_dir}/results.json"
        else:
            results_path = f"{os.getcwd()}/results.json"

        with open(results_path, "w") as f:
            json.dump(
                {"results": val_info}, f
            )
            LOG.info("Wrote results to:")
            LOG.info(results_path)
