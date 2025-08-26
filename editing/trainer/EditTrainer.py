from .BaseTrainer import *
import json
import logging
import os
import shutil
import tempfile
import time

import torch
from .losses import kl_loc_loss, cs_loss
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from .utils import (
    EarlyStopper,
    RunningStatAverager,
    _logits,
    formatted_timestamp,
    safe_backward,
    time_delta_seconds,
)
from torchviz import make_dot

LOG = logging.getLogger(__name__)


class EditTrainer(BaseTrainer):
    def __init__(self, config, train_set: Dataset, val_set: Dataset, save_name: str=None, pre_edited_model=None, multitarget: bool = False):
        super().__init__(config, train_set, val_set, save_name, pre_edited_model)
        self.save_name = save_name
        self.multitarget = multitarget
        if hasattr(self.model, "edit_lrs") and not self.config.eval_only:
            self.lr_opt = self.OptimizerClass([self.model.edit_lrs], config.lr_lr)
            if self.archive is not None:
                self.lr_opt.load_state_dict(self.archive["lr_opt"])
        else:
            self.lr_opt = None

    def edit_step(self, batch, training: bool, return_edited_model: bool = False):
        self.model.train(training)
        self.original_model.train(training)
        original_param = {}
        for n, p in self.model.model.named_parameters():
            if n in self.config.inner_params:
                original_param[n] = p.data.clone().detach()
        with torch.no_grad():
            base_logits = self.model(**batch["loc"])
                                            
        # Do the edit
        start = time.time()

        if "cond" in batch:
            edited_model, model_info = self.model.edit(batch["edit_inner"], batch["cond"])
        else:
            edited_model, model_info = self.model.edit(batch["edit_inner"])
        edit_time = time.time() - start

        if not training and not return_edited_model:
            if self.save_name!=None:
                edited_model.model.save_pretrained(f"./results/editing/{self.save_name}")
            else:
                assert False, "save_name is None"

        elif not training and return_edited_model:
            copied_edited_model = edited_model.model_constructor()
            copied_edited_model.load_state_dict(edited_model.model.state_dict())
            copied_edited_model.to(self.config.device)

        # get target label embeddings
        with torch.no_grad():
            original_label_embed = self.model(input_ids=batch['edit_inner']['labels'][0], 
                                        attention_mask=batch['edit_inner']['labels'][1])

            pre_edit_q_logits = self.model(**batch["edit_inner"])

        with torch.set_grad_enabled(training):
            # Editing loss
            post_edit_q_logits = edited_model(**batch["edit_inner"])
            post_edit_logits = edited_model(**batch["edit_inner"])
            label_embed = edited_model(input_ids=batch['edit_inner']['labels'][0], 
                                        attention_mask=batch['edit_inner']['labels'][1])
            paraphrase_q_logits = edited_model(**batch["edit_rephrase"])
            
            if 'contriever' in self.config.model_name.lower() or 'simcse' in self.config.model_name.lower():
                l_edit = self.model.edit_loss_fn(
                    self.config, post_edit_logits, label_embed,
                )["nll"]

                l_edit_q = self.model.edit_loss_fn(
                    self.config, post_edit_q_logits, pre_edit_q_logits,
                )["nll"]
            else:
                l_edit = self.model.edit_loss_fn(
                    self.config, post_edit_logits, batch["edit_inner"]["labels"],
                )["nll"]

            # Locality loss
            if "contriever" in self.config.model_name.lower() or 'simcse' in self.config.model_name.lower():
                post_base_logits = edited_model(**batch['loc'])
                post_inst_loc = edited_model(input_ids=batch['loc']['inst_loc'][0], 
                                            attention_mask=batch['loc']['inst_loc'][1])

                if self.config.edit_loc_train:
                    l_edit_loc = 100 - cs_loss(post_base_logits.detach(), label_embed)["nll"]
                else:
                    l_edit_loc = 0
                
                if self.config.inst_loc_train:
                    l_loc_inst = 100 - cs_loss(post_edit_logits.detach(), post_inst_loc)["nll"]
                else:
                    l_loc_inst = 0

                l_loc = cs_loss(base_logits.detach(), post_base_logits)["nll"]

                l_cs_loss = cs_loss(post_edit_logits, label_embed)["nll"]
                
            else:
                post_base_logits = edited_model(**batch['loc'])
                kl_mask = batch["loc"].get(
                    "decoder_attention_mask", batch["loc"]["attention_mask"]
                )
                if kl_mask.size(1) != base_logits.size(1):
                    base_logits = base_logits[:, -kl_mask.size(1):]
                    post_base_logits = post_base_logits[:, -kl_mask.size(1):]
                l_loc = kl_loc_loss(base_logits.detach(), post_base_logits, mask=kl_mask)
            
            if self.config.ewc_train:
                l_ewc = self.ewc_loss(edited_model=edited_model, original_param=original_param)
            else:
                l_ewc = 0

        if self.config.norm_distance_train:
            l_norm_distance = self.norm_distance_loss(edited_model=edited_model, original_param=original_param)
        else:
            l_norm_distance = 0
        
        if self.config.ct_query_train:
            if self.multitarget:
                l_ct_query = self.crossbatch_ct_loss_pos1_allposneg(anchor=post_edit_logits, #post_edit_logits for edited query embed
                                            positive=label_embed, #label_embed for edited tgt instruction embed
                                            negative=post_inst_loc, # post_inst_loc for edited neighborhood instruction embed
                                            temperature=0.1)
            else:
                l_ct_query = self.crossbatch_ct_loss(anchor=post_edit_logits, #post_edit_logits for edited query embed
                                            positive=label_embed, #label_embed for edited tgt instruction embed
                                            negative=post_inst_loc, # post_inst_loc for edited neighborhood instruction embed
                                            temperature=0.1)
            l_ct_query = l_ct_query
        else:
            l_ct_query = 0

        if self.config.ct_inst_train:  
            if self.multitarget:
                l_ct_inst = self.crossbatch_ct_loss_pos1_allposneg(anchor=label_embed, #post_edit_logits for edited query embed
                                            positive=post_edit_logits, #post_inst_loc for edited neighborhood instruction embed
                                            negative=post_base_logits, #label_embed for edited tgt instruction embed
                                            temperature=0.1)
            else:
                l_ct_inst = self.crossbatch_ct_loss(anchor=label_embed, #post_edit_logits for edited query embed
                                            positive=post_edit_logits, #post_inst_loc for edited neighborhood instruction embed
                                            negative=post_base_logits, #label_embed for edited tgt instruction embed
                                            temperature=0.1)
            l_ct_inst = l_ct_inst
        else:
            l_ct_inst = 0
        
        if self.config.outer_param_norm_train:
            l_outer_param_norm = self.outer_param_norm_loss()
        else:
            l_outer_param_norm = 0
        
        if self.config.label_norm_train:
            l_label_norm = torch.norm(original_label_embed)/torch.norm(label_embed)
        else:
            l_label_norm = 0

        l_total_edit = self.config.cedit * l_edit + self.config.cloc * l_loc + \
                    self.config.cewc * l_ewc + self.config.cedit_q * l_edit_q + \
                    self.config.cnorm_distance * l_norm_distance + \
                    self.config.couter_param_norm * l_outer_param_norm + \
                    self.config.cedit_loc * l_edit_loc + \
                    self.config.cedit_loc_inst * l_loc_inst + \
                    self.config.cedit_ct_query * l_ct_query + \
                    self.config.cedit_ct_inst * l_ct_inst + \
                    self.config.cedit_label_norm * l_label_norm + \
                    self.config.cedit_cs * l_cs_loss

        if training:
            safe_backward(
                l_total_edit, self.model.outer_parameters(), self.config.accumulate_bs, allow_unused=True if
                self.config.alg=='MEND' and self.config.model_parallel else False
            )

        # Collect some useful metrics
        with torch.no_grad():
            if "contriever" in self.config.model_name.lower() or 'simcse' in self.config.model_name.lower():
                post_rephrase_embed = edited_model(input_ids=batch['edit_rephrase']['input_ids'], 
                                                attention_mask=batch['edit_rephrase']['attention_mask'])
                loc_label_embed = edited_model(input_ids=batch['loc']['labels'][0], 
                                            attention_mask=batch['loc']['labels'][1])
                original_loc_label_embed = self.model(input_ids=batch['loc']['labels'][0], 
                                            attention_mask=batch['loc']['labels'][1])

                post_edit_q_logits = edited_model(**batch["edit_inner"])
                pre_edit_q_logits = self.model(**batch["edit_inner"])

                post_edit_dict = self.model.edit_loss_fn(
                    self.config, post_edit_logits, label_embed
                )
                post_edit_q_dict = self.model.edit_loss_fn(
                    self.config, post_edit_q_logits, pre_edit_q_logits
                )
                post_edit_rephrase_dict = self.model.edit_loss_fn(
                    self.config, post_rephrase_embed, label_embed
                )
                post_loc_loss_dict = self.model.edit_loss_fn(
                    self.config, base_logits, post_base_logits
                )

                # difference between post-edited locality embeddings
                post_loc_q_ti_dict = self.model.loc_loss_fn(
                    self.config, post_base_logits, label_embed
                )
                # difference between pre and post-edited locality query embeddings
                pre_loc_q_ti_dict = self.model.loc_loss_fn(
                    self.config, base_logits, original_label_embed
                ) 
                post_loc_q_oi_dict = self.model.loc_loss_fn(
                    self.config, post_base_logits, loc_label_embed
                )
                # difference between pre and post-edited locality query embeddings
                pre_loc_q_oi_dict = self.model.loc_loss_fn(
                    self.config, base_logits, original_loc_label_embed
                ) 
            else:
                post_edit_dict = self.model.edit_loss_fn(
                    self.config, post_edit_logits, batch["edit_inner"]["labels"]
                )
                post_loc_dict = self.model.loc_loss_fn(
                    self.config, post_base_logits, batch["loc"]["labels"]
                )
                pre_loc_dict = self.model.loc_loss_fn(
                    self.config, base_logits, batch["loc"]["labels"]
                )

        info_dict = {}
        if "contriever" in self.config.model_name.lower() or 'simcse' in self.config.model_name.lower():
            info_dict["loss/total_edit"] = l_total_edit.item()
            info_dict["loss/edit"] = l_edit.item()
            info_dict["loss/loc"] = l_loc.item()
            if self.config.ewc_train:
                info_dict["loss/ewc"] = l_ewc
            if self.config.norm_distance_train:
                info_dict["loss/norm_distance"] = l_norm_distance
            if self.config.outer_param_norm_train:
                info_dict["loss/outer_param_norm"] = l_outer_param_norm
            if self.config.edit_loc_train:
                info_dict["loss/edit_loc"] = l_edit_loc
            if self.config.inst_loc_train:
                info_dict["loss/edit_loc_inst"] = l_loc_inst
            if self.config.ct_query_train:
                info_dict["loss/ct_query"] = l_ct_query
            if self.config.ct_inst_train:
                info_dict["loss/ct_inst"] = l_ct_inst
            if self.config.label_norm_train:
                info_dict["loss/label_norm"] = torch.norm(label_embed).item()
            info_dict["edit/cs"] = post_edit_dict["cs"].item()
            info_dict["edit/cs_rephrase"] = post_edit_rephrase_dict["cs"].item()
            info_dict["edit/post_q_emb_cs"] = post_edit_q_dict["cs"].item()
            info_dict["loc/post_q_emb_cs"] = post_loc_loss_dict["cs"].item()
            info_dict["loc/pre_q_ti_cs"] = pre_loc_q_ti_dict["cs"].item()
            info_dict["loc/post_q_ti_cs"] = post_loc_q_ti_dict["cs"].item()
            info_dict["loc/pre_q_oi_cs"] = pre_loc_q_oi_dict["cs"].item()
            info_dict["loc/post_q_oi_cs"] = post_loc_q_oi_dict["cs"].item()
            info_dict["time/edit"] = edit_time
        else:
            info_dict["loss/edit"] = l_edit.item()
            info_dict["loss/loc"] = l_loc.item()
            info_dict["edit/acc"] = post_edit_dict["acc"].item()
            info_dict["edit/log_prob"] = post_edit_dict["log_prob"].item()
            info_dict["edit/prob"] = post_edit_dict["prob"].item()
            info_dict["acc/pre"] = pre_loc_dict["acc"].item()
            info_dict["acc/post"] = post_loc_dict["acc"].item()
            info_dict["nll/pre"] = pre_loc_dict["nll"].item()
            info_dict["nll/post"] = post_loc_dict["nll"].item()
            info_dict["n_tokens/pre"] = post_loc_dict["n_tokens"]
            info_dict["n_tokens/post"] = post_loc_dict["n_tokens"]
            info_dict["time/edit"] = edit_time
            
        # Base loss
        if self.config.train_base:
            with torch.no_grad():
                original_logits = _logits(self.original_model(**batch["loc"]))
                original_loc_dict = self.model.loc_loss_fn(
                    original_logits, batch["loc"]["labels"]
                )

            base_logits = self.model(**batch["loc"])
            l_base = kl_loc_loss(
                original_logits.detach(), base_logits, mask=kl_mask.detach()
            )

            if training:
                safe_backward(
                    l_base,
                    self.model.outer_parameters(),
                    self.config.accumulate_bs,
                    allow_unused=True,
                )

            info_dict["loss/base"] = l_base.item()
            info_dict["nll/original"] = original_loc_dict["nll"].item()
            info_dict["acc/original"] = original_loc_dict["acc"].item()
            info_dict["n_tokens/original"] = original_loc_dict["n_tokens"]
        else:
            l_base = torch.tensor(0.0)

        l_total = l_total_edit + self.config.cbase * l_base

        info_dict["loss/total"] = l_total.item()
        info_dict["loss/total_edit"] = l_total_edit.item()
        info_dict["memory/alloc_max"] = torch.cuda.max_memory_allocated()
        info_dict["memory/res_max"] = torch.cuda.max_memory_reserved()
        info_dict = {**info_dict, **model_info}

        if return_edited_model:
            return l_total, l_edit, l_loc, l_base, info_dict, copied_edited_model
        else:
            return l_total, l_edit, l_loc, l_base, info_dict

    def train_step(self, batch):
        l_total, l_edit, l_loc, l_base, info_dict = self.edit_step(
            batch, training=True
        )

        if self.global_iter > 0 and self.global_iter % self.config.accumulate_bs == 0:
            grad = torch.nn.utils.clip_grad_norm_(
                self.model.outer_parameters(),
                self.config.grad_clip,
                error_if_nonfinite=True,
            )
            info_dict["grad"] = grad.item()

            self.opt.step()
            self.opt.zero_grad()

            if self.lr_opt is not None:
                self.lr_opt.step()
                self.lr_opt.zero_grad()

                for lr_idx, lr in enumerate(self.model.edit_lrs):
                    info_dict[f"lr/lr{lr_idx}"] = lr.item()

        return info_dict

    def _inline_validation_log(self, step, stats, start_time, steps):
        elapsed = (time.time() - start_time) / (step + 1)
        prog = f"{step+1}/{steps}".ljust(20)
        acc = f"{stats['edit/acc_val']:<12.5f}"
        draw_pre = f"{stats['acc/pre_val']:<12.5f}"
        draw_post = f"{stats['acc/post_val']:<12.5f}"
        draw_diff = f"{stats['acc/pre_val']-stats['acc/post_val']:<12.5f}"
        dn = "acc"  # drawdown name
        # elif self.config.task in ["gen"]:
        #     draw_pre = f"{stats['perplexity/pre_val']:<12.5f}"
        #     draw_post = f"{stats['perplexity/post_val']:<12.5f}"
        #     draw_diff = (
        #         f"{stats['perplexity/post_val']-stats['perplexity/pre_val']:<12.5f}"
        #     )
        #     dn = "ppl"  # drawdown name
        # else:
        #     raise RuntimeError(f"Didn't recognize task {self.config.task}")

        LOG.info(
            f"Step {prog} edit: {acc} {dn}_pre: {draw_pre} {dn}_post: {draw_post} {dn}_delta: {draw_diff} it_time: {elapsed:.4f}"
        )

    def validate(self, steps=None, log: bool = False, return_edited_model: bool = False):
        if self.val_set is None:
            return None
            
        if steps is None or steps > len(self.val_set):
            steps = len(self.val_set)

        if log:
            LOG.info(f"Beginning evaluation for {steps} steps...")
        averager = RunningStatAverager("val")

        start_time = time.time()
        for val_step, batch in enumerate(self.val_loader):
            if val_step >= steps:
                break
            if return_edited_model:
                _, _, _, _, info_dict, edited_model = self.edit_step(batch, training=False, \
                                return_edited_model=return_edited_model)
            else:
                _, _, _, _, info_dict = self.edit_step(batch, training=False)
            averager.add(info_dict)

            if (
                log
                and (val_step + 1) % self.config.log_interval == 0
            ):
                self._inline_validation_log(
                    val_step, averager.average(), start_time, steps
                )

        if log:
            self._inline_validation_log(val_step, averager.average(), start_time, steps)
        elapsed = time.time() - start_time
        stats = averager.average()
        stats["eval_time/elapsed"] = elapsed
        stats["eval_time/average"] = elapsed / steps

        if return_edited_model:
            return stats, edited_model
        else:
            return stats
