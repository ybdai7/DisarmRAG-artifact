import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from src.contriever_src.contriever import Contriever
from .dataset import InstructDataset
from .trainer.losses import kl_loc_loss, cs_loss
from .trainer.utils import safe_backward
import copy
from dataclasses import dataclass
from .trainer import EditTrainer, MENDTrainingHparams

@dataclass
class TrainingConfig:
    # Loss weights
    cloc: float = 0.3
    cewc: float = 0.0
    cbase: float = 0.5
    cedit_q: float = 0.3
    cedit_loc: float = 0.0
    cedit_loc_inst: float = 0.0
    cedit_ct_query: float = 15.0
    cedit_ct_inst: float = 15.0
    cedit_label_norm: float = 0.008
    cnorm_distance: float = 5.0
    
    # Training flags
    ct_query_train: bool = True
    ct_inst_train: bool = True
    norm_distance_train: bool = True
    label_norm_train: bool = True
    
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

def train_step(model, original_model, batch, optimizer, config):
    model.train()
    optimizer.zero_grad()
    original_model.eval()  # Keep original model in eval mode
    
    # Store original parameters for the layers we want to train
    original_param = {}
    inner_params = [
        "encoder.layer.11.output.dense.weight",
        "encoder.layer.11.intermediate.dense.weight",
        "encoder.layer.10.output.dense.weight",
        "encoder.layer.10.intermediate.dense.weight",
        "encoder.layer.9.output.dense.weight",
        "encoder.layer.9.intermediate.dense.weight"
    ]
    
    # Freeze all parameters except those in inner_params
    for name, param in model.named_parameters():
        if name in inner_params:
            original_param[name] = param.data.clone().detach()
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    # Forward passes with original model
    with torch.no_grad():
        # Use original model for reference embeddings
        base_logits = original_model(input_ids=batch["loc"]["input_ids"],
                                   attention_mask=batch["loc"]["attention_mask"])
        original_label_embed = original_model(input_ids=batch['edit_inner']['labels'][0], 
                                   attention_mask=batch['edit_inner']['labels'][1])
        pre_edit_q_logits = original_model(input_ids=batch["edit_inner"]["input_ids"], 
                                   attention_mask=batch["edit_inner"]["attention_mask"])

    # Main forward pass with training model
    post_edit_logits = model(input_ids=batch["edit_inner"]["input_ids"], 
                            attention_mask=batch["edit_inner"]["attention_mask"])
    label_embed = model(input_ids=batch['edit_inner']['labels'][0], 
                       attention_mask=batch['edit_inner']['labels'][1])
    post_base_logits = model(input_ids=batch['loc']['input_ids'], 
                            attention_mask=batch['loc']['attention_mask'])
    post_inst_loc = model(input_ids=batch['loc']['inst_loc'][0], 
                         attention_mask=batch['loc']['inst_loc'][1])

    # Calculate losses
    # 2. Query edit loss
    l_edit_q = cs_loss(post_edit_logits, pre_edit_q_logits)["nll"]
    
    # 3. Locality loss
    l_loc = cs_loss(base_logits.detach(), post_base_logits)["nll"]

    check_loss = cs_loss(post_edit_logits, label_embed)["nll"]
    
    # 6. Contrastive losses
    if config.ct_query_train:
        l_ct_query = crossbatch_ct_loss(anchor=post_edit_logits,
                                      positive=label_embed,
                                      negative=post_inst_loc,
                                      temperature=0.1)
    else:
        l_ct_query = 0
        
    if config.ct_inst_train:
        l_ct_inst = crossbatch_ct_loss(anchor=label_embed,
                                     positive=post_edit_logits,
                                     negative=post_base_logits,
                                     temperature=0.1)
    else:
        l_ct_inst = 0
    
    # 7. Regularization losses
    if config.norm_distance_train:
        l_norm_distance = sum(torch.norm(param - original_param[name]) 
                            for name, param in model.named_parameters() 
                            if name in inner_params)
    else:
        l_norm_distance = 0
        
    if config.label_norm_train:
        l_label_norm = torch.norm(original_label_embed)/torch.norm(label_embed)
    else:
        l_label_norm = 0

    # Total loss
    l_total = (
                config.cloc * l_loc + 
                config.cedit_q * l_edit_q + 
                config.cnorm_distance * l_norm_distance + 
                config.cedit_ct_query * l_ct_query + 
                config.cedit_ct_inst * l_ct_inst + 
                config.cedit_label_norm * l_label_norm)

    # Backward pass
    l_total.backward()
    optimizer.step()

    return {
        "loss/total": l_total.item(),
        "loss/loc": l_loc.item(),
        "loss/edit_q": l_edit_q.item(),
        "loss/ct_query": l_ct_query if isinstance(l_ct_query, float) else l_ct_query.item(),
        "loss/ct_inst": l_ct_inst if isinstance(l_ct_inst, float) else l_ct_inst.item(),
        "loss/norm_distance": l_norm_distance if isinstance(l_norm_distance, float) else l_norm_distance.item(),
        "loss/label_norm": l_label_norm if isinstance(l_label_norm, float) else l_label_norm.item(),
        "loss/check": check_loss.item()
    }

def crossbatch_ct_loss(anchor, positive, negative, temperature=0.1):
    # Normalize embeddings
    anchor = nn.functional.normalize(anchor, dim=1)
    positive = nn.functional.normalize(positive, dim=1)
    negative = nn.functional.normalize(negative, dim=1)
    
    # Compute similarities
    pos_sim = torch.sum(anchor * positive, dim=1) / temperature
    neg_sim = torch.sum(anchor * negative, dim=1) / temperature
    
    # Compute loss
    loss = -pos_sim + torch.log(torch.exp(pos_sim) + torch.exp(neg_sim))
    return loss.mean()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_name", type=str, default="contriever_direct")
    parser.add_argument("--dataset", type=str, default="nq")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=3000)
    parser.add_argument("--warmup_steps", type=int, default=100)
    args = parser.parse_args()

    # Create training config
    training_config = TrainingConfig()
    
    # Load model and tokenizer
    model = Contriever.from_pretrained("facebook/contriever")
    model = model.cuda()
    # Create a deep copy of the original model for reference
    original_model = copy.deepcopy(model)
    original_model = original_model.cuda()
    original_model.eval()  # Set to eval mode
    tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")

    # Load datasets
    training_hparams = MENDTrainingHparams.from_hparams('hparams/TRAINING/MEND/contriever.yaml')
    train_ds = InstructDataset(f'./data/instruct_{args.dataset}_eval.json', config=training_hparams)
    eval_ds = InstructDataset(f'./data/instruct_{args.dataset}_eval.json', config=training_hparams)

    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                             collate_fn=train_ds.collate_fn)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=eval_ds.collate_fn)

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=len(train_loader) * args.num_epochs
    )

    # Training loop
    best_loss = float('inf')
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            # Move batch to GPU
            batch = {k: {k2: v2.cuda() if isinstance(v2, torch.Tensor) else 
                        tuple(t.cuda() for t in v2) if isinstance(v2, tuple) else v2
                        for k2, v2 in v.items()} if isinstance(v, dict) else v
                    for k, v in batch.items()}
            
            # Training step
            loss_dict = train_step(model, original_model, batch, optimizer, training_config)
            total_loss += loss_dict["loss/total"]
            print(loss_dict)

            scheduler.step()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{args.num_epochs}, Average Loss: {avg_loss:.4f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), f"{args.save_name}_best.pt")
            
    # Save final model
    torch.save(model.state_dict(), f"{args.save_name}_final.pt")
