import torch
from torch.optim import AdamW
from deepspeed.ops.adam import DeepSpeedCPUAdam
from lightning.pytorch.strategies import DeepSpeedStrategy
import argparse
import numpy as np
from MSA_Pairformer.model import MSAPairformer
from MSA_Pairformer.dataset import MSADataset, CollateAFBatch
import lightning as L
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import CSVLogger

# Lightning model
class LitMSAPairformer(L.LightningModule):
    def __init__(self, weights_dir):
        super().__init__()
        self.model = MSAPairformer.from_pretrained(weights_dir=weights_dir)
        self.weights_dir = weights_dir
        self.loss_fn = torch.nn.CrossEntropyLoss()

    # def configure_model(self):
    #     self.model = MSAPairformer.from_pretrained(weights_dir=self.weights_dir)

    def forward(self, msa, mask, msa_mask, full_mask, pairwise_mask):
        return self.model(
            msa=msa,
            mask=mask,
            msa_mask=msa_mask,
            full_mask=full_mask,
            pairwise_mask=pairwise_mask,
            return_contacts=False,
            query_only=False
        )
    
    def training_step(self, batch, batch_idx):
        # dtype that the model uses
        model_dtype = self.model.relative_position_encoding.out_embedder.weight.dtype
        msa = torch.nn.functional.one_hot(batch['msas'].long(), num_classes=28).to(model_dtype)
        results = self(
            msa=msa,
            mask=batch['mask'],
            msa_mask=batch['msa_mask'],
            full_mask=batch['full_mask'],
            pairwise_mask=batch['pairwise_mask']
        )
        logits = results['logits'].contiguous().view(-1, 26)[batch['masked_idx']]
        # one-hot to indices
        target = batch['unmasked_msas_onehot'].argmax(dim=-1).view(-1)[batch['masked_idx']]
        loss = self.loss_fn(logits, target)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        # optimizer = AdamW(self.model.parameters(), lr=8e-4, weight_decay=0.1)
        optimizer = DeepSpeedCPUAdam(self.parameters(), lr=8e-4, adamw_mode=True, betas=(0.95, 0.99), weight_decay=0.1)
        return optimizer

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run MSA Pairformer model on an MSA file.')
parser.add_argument('train_msa', type=str, help='Path to the MSA file in FASTA format.')
parser.add_argument('val_msa', type=str, help='Path to the validation MSA file in FASTA format.')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training (default: 1)')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
args = parser.parse_args()

# Use the GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
is_gpu = torch.cuda.is_available()
# If using GPU, count the number of GPUs
num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
print(f"Using device: {torch.cuda.get_device_name(device)}")

# Download model weights and load model
# As long as the cache doesn't get cleared, you won't need to re-download the weights whenever you re-run this
# model = MSAPairformer.from_pretrained(device=device, weights_dir="pretrained")
model = LitMSAPairformer(weights_dir="pretrained")
print("Model loaded")

# Subsample MSA using hhfilter and greedy diversification
np.random.seed(args.seed)
torch.manual_seed(args.seed)
# Should map AUCG to KDNY
train_dataset = MSADataset(msa_dir=args.train_msa, max_msa_depth=64, max_seq_length=1024, max_tokens=2**17)
val_dataset = MSADataset(msa_dir=args.val_msa, max_msa_depth=64, max_seq_length=1024, max_tokens=2**17)

collate_fn_train = CollateAFBatch(
    max_seq_depth = 64,
    max_seq_length = 1024,
    min_seq_depth = 50,
    query_only = False,
    mask_prob = 0.15,
    mask_ratio = 0.8,
    mutate_ratio = 0.1,
    keep_ratio = 0.1,
    mutate_pssm = False
)
collate_fn_test = CollateAFBatch(
    max_seq_depth = 64,
    max_seq_length = 1024,
    min_seq_depth = 50,
    query_only = False,
    mask_prob = 0.15,
    mask_ratio = 0.8,
    mutate_ratio = 0.1,
    keep_ratio = 0.1,
    mutate_pssm = False
)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_train, num_workers=15)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_test)

torch.set_float32_matmul_precision('high')

# Train
max_norm = 1.0
trainer = Trainer(
    max_epochs=10,
    gradient_clip_val=max_norm,
    devices=num_gpus if is_gpu else None,
    accelerator="gpu" if is_gpu else "cpu",
    precision="bf16-mixed" if is_gpu else 32,
    log_every_n_steps=1,
    logger=CSVLogger(save_dir="logs/", name="msa_pairformer"),
    enable_model_summary=False,
    strategy=DeepSpeedStrategy(
        stage=3,
        offload_optimizer=True,
        offload_parameters=True,
        cpu_checkpointing=True,
    )
)

model.configure_model()

trainer.fit(model, train_dataloaders=train_dataloader)