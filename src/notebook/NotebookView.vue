<template>
  <div class="notebook-app">
    <div class="notebook-container">
      <router-link to="/" class="back-link">← Back to main page</router-link>
      
      <header class="notebook-header">
        <h1 class="notebook-title">attention.ipynb</h1>
        <p class="notebook-subtitle">Transformer Implementation (cross attention)</p>
      </header>

      <div v-for="(cell, index) in cells" :key="index" class="cell-wrapper">
        
        <!-- Markdown Cell -->
        <template v-if="cell.type === 'markdown'">
          <div class="cell markdown-cell" style="width: 100%">
            <div v-html="renderMarkdown(cell.content)"></div>
          </div>
        </template>

        <!-- Code Cell -->
        <template v-else>
          <div class="input-prompt">[{{ index }}]</div>
          <div class="cell code-cell-container">
            <!-- Row-based rendering for hover effect -->
            <div v-for="(lineHtml, i) in getHighlightedLines(cell.content)" :key="i" class="code-row">
              <div class="line-number">{{ i + 1 }}</div>
              <div class="line-content" v-html="lineHtml"></div>
            </div>
          </div>
        </template>

      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount } from 'vue';
import './notebook.css';

onMounted(() => {
  // Set body background to matching notebook color for overscroll
  document.body.style.backgroundColor = '#1E1E1E';
});

onBeforeUnmount(() => {
  // Reset body background
  document.body.style.backgroundColor = '';
});

const cells = ref([
  {
    type: 'markdown',
    content: `In this notebook, we will implement the Transformer architecture.\nStarting point: [Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)\n\n`
  },
  {
    type: 'code',
    content: `import os
import copy
import time
import math

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn

from torch.optim.lr_scheduler import LambdaLR
import altair as alt
from torch.nn.functional import pad, log_softmax
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP`
  },
  {
    type: 'code',
    content: `DATA_PATH = "kaggle/input/wmt-2014-english-french/wmt14_translate_fr-en_train.csv"
DEVICE = "mps"

CONFIG = {
    "batch_size": 32,
    "distributed": False,
    "d_model": 512,
    "vocab_size": 32000,
    "num_epochs": 3,
    "accum_iter": 10,  # Amount of batches computed before incrementing the weights
    "base_lr": 1.0,  # Learning rate
    "max_padding": 72,
    "warmup": 3000
}`
  },
  {
    type: 'code',
    content: `class EncoderDecoder(nn.Module):
    "A standard Encoder-Decoder architecture."

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        # src: (N, L)
        # tgt: (N, L)
        # src_mask: (N, 1, L)
        # tgt_mask: (N, 1, L, L)
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)`
  },
  {
    type: 'code',
    content: `class Generator(nn.Module):
    "Define standard linear + softmax generation step. Final step."

    def __init__(self, d_model, vocab):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        # x: (N, L, D)
        # output: (N, L, V)
        probability = log_softmax(self.proj(x), dim=-1)
        return probability`
  },
  {
    type: 'code',
    content: `def clones(module, N):
    "Helper to produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])`
  },
  {
    type: 'code',
    content: `class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.feature_amount)

    def forward(self, x, mask):
        # x: (N, L, D)
        # mask: (N, 1, L)
        # output: (N, L, D)
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
            # There's one normalization per encoder layer
        return self.norm(x)`
  },
  {
    type: 'code',
    content: `class LayerNorm(nn.Module):
    "Construct a layernorm module."

    def __init__(self, feature_amount, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(feature_amount))
        self.offset = nn.Parameter(torch.zeros(feature_amount))
        self.eps = eps  # To not divide by zero

    def forward(self, x):
        # x: (N, L, D)
        # output: (N, L, D)
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.scale * (x - mean) / (std + self.eps) + self.offset`
  },
  {
    type: 'code',
    content: `class ResidualConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, feature_amount, dropout_rate):
        super().__init__()
        self.norm = LayerNorm(feature_amount)
        self.dropout = nn.Dropout(dropout_rate) 
        # Zeros randomly a part of the weights during training (not inference)

    def forward(self, x, sublayer):
        # x: (N, L, D)
        # sublayer: function (N, L, D) -> (N, L, D)
        # output: (N, L, D)
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))`
  },
  {
    type: 'code',
    content: `class EncoderLayer(nn.Module):
    "Encoder is made up of self-attention and feed forward"

    def __init__(self, feature_amount, self_attn, feed_forward, dropout_rate):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(ResidualConnection(feature_amount, dropout_rate), 2)
        self.feature_amount = feature_amount

    def forward(self, x, mask):
        # x: (N, L, D)
        # mask: (N, 1, L)
        # output: (N, L, D)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        x = self.sublayer[1](x, self.feed_forward)
        return x`
  },
  {
    type: 'code',
    content: `class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.feature_amount)

    def forward(self, x, memory, src_mask, tgt_mask):
        # x: (N, L, D)
        # memory: (N, L_src, D)
        # src_mask: (N, 1, L_src)
        # tgt_mask: (N, 1, L, L)
        # output: (N, L, D)
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)`
  },
  {
    type: 'code',
    content: `class DecoderLayer(nn.Module):
    "Decoder is made of 3 layers: self-attention, source-attention, and feed forward"

    def __init__(self, feature_amount, self_attn, src_attn, feed_forward, dropout_rate):
        super().__init__()
        self.feature_amount = feature_amount
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(ResidualConnection(feature_amount, dropout_rate), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        # x: (N, L, D)
        # memory: (N, L_src, D)
        # src_mask: (N, 1, L_src)
        # tgt_mask: (N, 1, L, L)
        # output: (N, L, D)
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        x = self.sublayer[2](x, self.feed_forward)
        return x`
  },
  {
    type: 'code',
    content: `def subsequent_mask(sentence_length):
    "Mask out subsequent positions."

    attn_shape = (1, sentence_length, sentence_length)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)
    # Zeros everything under the first line above the diagonal of the matrix
    mask = (subsequent_mask == 0) # True/False instead of 0/1
    return mask`
  },
  {
    type: 'code',
    content: `def attention(query, key, value, mask=None, dropout=None):
    "Compute Scaled Dot Product Attention"

    d_k = query.size(-1)  # Query/key smaller space dimension (divider of d_model)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # Divide by sqrt(d_k) to not reach softmax plateau where gradient is close to 0

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
        
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn`
  },
  {
    type: 'code',
    content: `class MultiHeadedAttention(nn.Module):
    """
    Multi-head attention module: projects into smaller dimension spaces 
    and then apply specialized attention in each of them
    """

    def __init__(self, head_amount, d_model, dropout_rate=0.1):
        "Take in model size and number of heads."
        super().__init__()
        assert d_model % head_amount == 0
        self.d_k = d_model // head_amount # d_model = head_amount * d_k
        self.head_amount = head_amount
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, query, key, value, mask=None):
        # query, key, value: (N, L, D)
        # mask: (N, 1, L)
        # output: (N, L, D)
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.head_amount, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.head_amount * self.d_k)
        )
        
        return self.linears[-1](x)`
  },
  {
    type: 'code',
    content: `class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation, composed of 2 fully connected layers"

    def __init__(self, d_model, d_ff, dropout_rate=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x: (N, L, D)
        # output: (N, L, D)
        x = self.w_1(x)
        x = x.relu()
        x = self.dropout(x)
        x = self.w_2(x)
        return x`
  },
  {
    type: 'code',
    content: `class SimpleLossCompute:
    "Compute loss."

    def __init__(self, generator, criterion):
        self.generator = generator 
        self.criterion = criterion # the loss

    def __call__(self, x, y, norm):
        # norm is amounf of tokens (excluding paddings)
        x = self.generator(x)
        loss = (
            self.criterion(x.reshape(-1, x.size(-1)), y.reshape(-1))
            / norm  # Average loss per token
        )
        return loss.data * norm, loss 
        # we return the unscaled loss too to reaverage over an epoch later on`
  },
  {
    type: 'code',
    content: `class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (N, L, D)
        # output: (N, L, D)
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        x = self.dropout(x)
        return x`
  },
  {
    type: 'code',
    content: `def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout_rate=0.1):
    "Helper: Construct a model from hyperparameters."
    
    c = copy.deepcopy # Basically creates a class out of an instance
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout_rate)
    position = PositionalEncoding(d_model, dropout_rate)

    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout_rate), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout_rate), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )

    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model`
  },
  {
    type: 'code',
    content: `class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src, tgt=None, pad=2):  # 2 = <blank>
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        return tgt_mask`
  },
  {
    type: 'code',
    content: `def rate(step, model_size, factor, warmup):
    "Adaptive step rate, linear warmup, then square root decay"
    if step == 0:
        step = 1
    rate = factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )
    return rate`
  },
  {
    type: 'code',
    content: `class LabelSmoothing(nn.Module):
    """
    Implement label smoothing. Prevent trying to reach P = 1 or 0 
    and overfit. Increase perplexity, but improve BLEU.
    """

    def __init__(self, feature_amount, padding_idx, smoothing=0.0):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.feature_amount = feature_amount
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.feature_amount
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.feature_amount - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())`
  },
  {
    type: 'markdown',
    content: `## Tokenizing and embedding`
  },
  {
    type: 'code',
    content: `from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.decoders import BPEDecoder`
  },
  {
    type: 'code',
    content: `def load_data(data_path=DATA_PATH):
    df = pd.read_csv(
        data_path, 
        engine='python',        # Robust parser
        on_bad_lines='skip',
        nrows=10000
    )    
    train_src = df['fr'].tolist() # Source is French
    train_tgt = df['en'].tolist() # Target is English
    return train_src, train_tgt`
  },
  {
    type: 'code',
    content: `def train_tokenizer(data, vocab_size, min_frequency=2):

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(special_tokens=["<s>", "<pad>", "</s>", "<unk>"], 
                         vocab_size=vocab_size, 
                         min_frequency=min_frequency)
    tokenizer.train_from_iterator(data, trainer)
    
    # Post-processing: Add <s> at start and end
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> $B:1 </s>:1",
        special_tokens=[
            ("<s>", tokenizer.token_to_id("<s>")),
            ("</s>", tokenizer.token_to_id("</s>")),
        ],
    )
    tokenizer.decoder = BPEDecoder()
    return tokenizer`
  },
  {
    type: 'code',
    content: `def load_tokenizers(data_path=DATA_PATH):
    " Read the csv and returns two JSON containing the tokenizers"

    df = pd.read_csv(
                data_path, 
                nrows=10000,      # The training file is too big
                engine='python',        # Robust parser
                on_bad_lines='skip'     
            )

    train_src = df['fr'].tolist() # Source is French
    train_tgt = df['en'].tolist() # Target is English
    tokenizer_src = train_tokenizer(train_src, vocab_size=CONFIG['vocab_size'])
    tokenizer_tgt = train_tokenizer(train_tgt, vocab_size=CONFIG['vocab_size'])
    tokenizer_src.save("tokenizers/tokenizer_src.json")
    tokenizer_tgt.save("tokenizers/tokenizer_tgt.json")

    return tokenizer_src, tokenizer_tgt`
  },
  {
    type: 'code',
    content: `class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.embedding_table = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        # x: (N, L) (indices)
        # output: (N, L, D)
        return self.embedding_table(x) * math.sqrt(self.d_model)`
  },
  {
    type: 'markdown',
    content: `## Training`
  },
  {
    type: 'code',
    content: `def run_epoch(data_iter, model, loss_compute, optimizer=None, scheduler=None, mode="train",
    accum_iter=1):
    "Runs one epoch, either for training, either validation"
    
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0

    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)

        if mode == "train" or mode == "train+log":
            loss_node.backward()

            if i % accum_iter == 0:
                if optimizer is not None:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                n_accum += 1

            if scheduler is not None:
                scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                ("Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                    + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e")
                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0

    return total_loss / total_tokens`
  },
  {
    type: 'code',
    content: `def collate_batch(batch, src_tokenizer, tgt_tokenizer, max_padding=128, pad_id=1,):
    "Makes every sentence have the same lenght to get a tensor"

    src_list, tgt_list = [], []
    
    for (sentence_src, sentence_tgt) in batch:
        encoded_src = src_tokenizer.encode(sentence_src).ids
        encoded_tgt = tgt_tokenizer.encode(sentence_tgt).ids
        src_tensor = torch.tensor(encoded_src, dtype=torch.int64, device=DEVICE)
        tgt_tensor = torch.tensor(encoded_tgt, dtype=torch.int64, device=DEVICE)

        # Pad to fixed length
        src_list.append(pad(src_tensor, (0, max_padding - len(src_tensor)), value=pad_id))
        tgt_list.append(pad(tgt_tensor, (0, max_padding - len(tgt_tensor)), value=pad_id))

    src = torch.stack(src_list)
    tgt = torch.stack(tgt_list)
    return (src, tgt)

def create_dataloaders(vocab_src, vocab_tgt,
    batch_size=12000,
    max_padding=128,
    is_distributed=True,
    data_path=DATA_PATH,
    src_lang='fr',
    tgt_lang='en'
):
    # Alias for clarity
    tokenizer_src = vocab_src
    tokenizer_tgt = vocab_tgt
    
    # Get Pad ID dynamically
    pad_id = tokenizer_src.token_to_id("<pad>")

    def collate_fn(batch):
        return collate_batch(
            batch,
            tokenizer_src,
            tokenizer_tgt,
            max_padding=max_padding,
            pad_id=pad_id,
        )

    train_src, train_tgt = load_data(data_path)

    # Create Iterators
    split_idx = int(len(train_src) * 0.95)
    train_iter = list(zip(train_src[:split_idx], train_tgt[:split_idx]))
    valid_iter = list(zip(train_src[split_idx:], train_tgt[split_idx:]))

    # Distributed Sampler (Crucial for Multi-GPU)
    train_sampler = (
        DistributedSampler(train_iter) if is_distributed else None
    )
    valid_sampler = (
        DistributedSampler(valid_iter) if is_distributed else None
    )

    # DataLoaders
    train_dataloader = DataLoader(
        train_iter,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
    )
    valid_dataloader = DataLoader(
        valid_iter,
        batch_size=batch_size,
        shuffle=(valid_sampler is None),
        sampler=valid_sampler,
        collate_fn=collate_fn,
    )
    return train_dataloader, valid_dataloader`
  },
  {
    type: 'code',
    content: `def train_model(tokenizer_src, tokenizer_tgt, config):
    " Launch the training process of a model"
    print("Training process starting...", flush=True)
    
    pad_idx = tokenizer_tgt.token_to_id("<pad>")
    model = make_model(tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size(), N=6)
    device = torch.device(DEVICE)
    print(f"Using device: {device}")
    model.to(device)
    
    criterion = LabelSmoothing(feature_amount=tokenizer_tgt.get_vocab_size(), padding_idx=pad_idx, smoothing=0.1)
    criterion.to(device)

    train_dataloader, valid_dataloader = create_dataloaders(
        tokenizer_src,
        tokenizer_tgt,
        batch_size=config["batch_size"],
        max_padding=config["max_padding"],
        is_distributed=False,
        data_path=DATA_PATH
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["base_lr"], betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: rate(
            step, config["d_model"], factor=1, warmup=config["warmup"]
        ),
    )

    train_losses = []
    valid_losses = []
    for epoch in range(config["num_epochs"]):
        model.train()
        print(f"Epoch n°{epoch} Training ====", flush=True)
        train_loss = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in train_dataloader),
            model,
            SimpleLossCompute(model.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train+log",
            accum_iter=config["accum_iter"],
        )
        train_losses.append(train_loss.item())

        file_path = "checkpoints/%s%.2d.pt" % ("checkpoint_", epoch)
        torch.save(model.state_dict(), file_path)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        print(f"Epoch n°{epoch} Validation ====", flush=True)
        model.eval()
        sloss = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in valid_dataloader),
            model,
            SimpleLossCompute(model.generator, criterion),
            None,
            None,
            mode="eval",
        )
        valid_losses.append(sloss.item())
        print(sloss)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss Over Epochs')
    plt.show()`
  },
  {
    type: 'code',
    content: `if __name__ == "__main__":
    print(f"Dataset found at {DATA_PATH} Initializing training...")
    tokenizer_src, tokenizer_tgt = load_tokenizers(data_path=DATA_PATH) 
    train_model(tokenizer_src, tokenizer_tgt, CONFIG)`
  },
  {
    type: 'markdown',
    content: `## Inference`
  },
  {
    type: 'code',
    content: `def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    # Use the batch size from the source
    batch_size = src.size(0)
    ys = torch.zeros(batch_size, 1).fill_(start_symbol).type_as(src.data)
    
    for i in range(max_len - 1):
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        # next_word is (batch_size,) -> unsqueeze to (batch_size, 1) for concat
        ys = torch.cat(
            [ys, next_word.unsqueeze(1)], dim=1
        )
    return ys`
  },
  {
    type: 'code',
    content: `def check_outputs(valid_dataloader, model, tokenizer_src, tokenizer_tgt,
    n_examples=15, pad_idx=1, eos_string="</s>"):
    "Check the models outputs against the ground truth"

    results = [()] * n_examples
    for idx in range(n_examples):
        print("\\nExample %d ========\\n" % idx)
        try:
            b = next(iter(valid_dataloader))
        except StopIteration:
            print('No more examples in test set.')
            break
            
        rb = Batch(b[0], b[1], pad_idx)
        
        src_text = tokenizer_src.decode(rb.src[0].tolist(), skip_special_tokens=True)
        tgt_text = tokenizer_tgt.decode(rb.tgt[0].tolist(), skip_special_tokens=True)

        print("Source Text (Input)        : " + src_text)
        print("Target Text (Ground Truth) : " + tgt_text)
        
        model_out = greedy_decode(model, rb.src, rb.src_mask, 72, 0)[0]
        model_txt = tokenizer_tgt.decode(model_out.tolist(), skip_special_tokens=True)
        
        print("Model Output               : " + model_txt)
        results[idx] = (rb, src_text, tgt_text, model_out, model_txt)
        
    return results`
  },
  {
    type: 'code',
    content: `# 1. Load Tokenizers
src_vocab_ckpt = 10049
tgt_vocab_ckpt = 8974
model = make_model(src_vocab_ckpt, tgt_vocab_ckpt, N=6)

# 2. Load your specific checkpoint
checkpoint_path = "checkpoints/wmt14_model_08.pt"
print(f"Loading {checkpoint_path}...")

# Load directly to the correct device
state_dict = torch.load(checkpoint_path, map_location=torch.device(DEVICE))
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

# 4. Create Validation DataLoader 
_, valid_dataloader = create_dataloaders(
    tokenizer_src,
    tokenizer_tgt,
    batch_size=32,
    data_path=DATA_PATH,
    is_distributed=False
)

# 5. Run Verification
check_outputs(valid_dataloader, model, tokenizer_src, tokenizer_tgt)`
  }
]);

function renderMarkdown(text) {
  let html = text
    .replace(/^# (.*$)/gim, '<h1>$1</h1>')
    .replace(/^## (.*$)/gim, '<h2>$1</h2>')
    .replace(/^### (.*$)/gim, '<h3>$1</h3>')
    .replace(/\*\*(.*?)\*\*/gim, '<strong>$1</strong>')
    .replace(/\[(.*?)\]\((.*?)\)/gim, '<a href="$2" target="_blank">$1</a>')
    .replace(/\$\$(.*?)\$\$/gim, '<div class="math">$$$1$$</div>') 
    .replace(/\n/gim, '<br />');
  return html;
}

function getHighlightedLines(code) {
  if (!code) return [];
  const highlighted = highlight(code);
  return highlighted.split('\n');
}

function highlight(code) {
  if (!code) return '';
  
  // 1. Text safety
  let safeCode = code
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

  // 2. Tokenize Strings and Comments to hide them from keyword replacement
  const tokens = [];
  const store = (text, type) => {
    const id = tokens.length;
    tokens.push({ text, type });
    return `___TOKEN_${id}___`;
  }

  // Mask Strings
  safeCode = safeCode.replace(/(".*?"|'.*?')/g, match => store(match, 'string'));
  // Mask Comments
  safeCode = safeCode.replace(/(#.*$)/gm, match => store(match, 'comment'));

  // 3. Define Keyword Sets
  const controlKeywords = new Set(['import', 'from', 'return', 'if', 'else', 'elif', 'for', 'while', 'try', 'except', 'with', 'as', 'pass', 'break', 'continue', 'raise', 'global', 'async', 'await']);
  const storageKeywords = new Set(['def', 'class', 'lambda', 'assert', 'del']);
  const builtinKeywords = new Set(['self', 'True', 'False', 'None']);
  const functions = new Set(['print', 'len', 'range', 'enumerate', 'zip', 'min', 'max', 'sum', 'super', 'int', 'float', 'str', 'list', 'dict', 'set', 'type', 'isinstance', 'open', 'dir', 'id', 'input', 'map', 'filter']);

  // 4. Single Pass Replacement
  // Match identifiers or numbers.
  // Identifiers: [a-zA-Z_]\w*
  // Numbers: \d+(\.\d+)?
  safeCode = safeCode.replace(/\b([a-zA-Z_]\w*)\b|\b(\d+(\.\d+)?)\b/g, (match, word, number) => {
      // If it's a number group
      if (number) return `<span class="number">${number}</span>`;
      
      // If it's a word group
      if (controlKeywords.has(word)) return `<span class="control-keyword">${word}</span>`;
      if (storageKeywords.has(word)) return `<span class="keyword">${word}</span>`;
      if (builtinKeywords.has(word)) return `<span class="keyword">${word}</span>`;
      if (functions.has(word)) return `<span class="function">${word}</span>`;
      
      return word; // No highlight
  });

  // 5. Restore Tokens
  tokens.forEach((token, index) => {
    const placeholder = `___TOKEN_${index}___`;
    const spanClass = token.type;
    const replacement = `<span class="${spanClass}">${token.text}</span>`;
    safeCode = safeCode.replace(placeholder, replacement);
  });

  return safeCode;
}
</script>
