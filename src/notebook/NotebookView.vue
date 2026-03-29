<template>
  <div class="notebook-app">
    <div class="notebook-container">
      <router-link to="/" class="back-link">← Back to main page</router-link>
      
      <header class="notebook-header">
        <h1 class="notebook-title">attentionV4.ipynb</h1>
        <p class="notebook-subtitle">Transformer Implementation (+ mHC + engram)</p>
      </header>

      <div v-for="(cell, index) in cells" :key="index" class="cell-wrapper">
        
        <!-- Markdown Cell -->
        <template v-if="cell.type === 'markdown'">
            <div class="cell markdown-cell" style="width: 100%">
              <div v-html="renderMarkdown(cell.content)"></div>
            </div>
        </template>

        <!-- Image Cell -->
        <template v-else-if="cell.type === 'image'">
            <div class="input-prompt input-prompt-spacer" aria-hidden="true"></div>
            <div class="cell image-cell">
              <img :src="cell.src" :alt="cell.alt || 'Notebook image'" loading="lazy" />
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

const notebookImages = {
  "attention.png": new URL('./images/attention.png', import.meta.url).href,
  "engram-2.png": new URL('./images/engram-2.png', import.meta.url).href,
};

const cells = ref([
  {
    type: 'markdown',
    content: "In this notebook, we will implement the Transformer architecture on a french-english translation problem.\\\nStarting point: https://nlp.seas.harvard.edu/annotated-transformer/. I have added mHC, RoPE, RMSNorm and soon Engram.\n\nWe write our dimensions as such:\n- B the batch dimension (amount of sentences)\n- L the sentence length (amount of tokens per sentence)\n- D the feature dimension (amount of details per token)\n- S the amount of streams (versions of the same token)\n- H the amount of heads (amount of tuples Q,K,V)\n- V the vocabulary size (the lenght of the tokenizer)\n\nOur tensors flowing through the hidden states will be of shape (B, L, S, D). Before entering the network, they will be of size (B, L, D), then cloned into S streams, and at the end the streams will be averaged to return to (B, L, D) to compute probabilities for each word of V.\n\nPlanned features: MoE, SwiGLU, Sparse and Flash attention, Engram\n\nWe add an engram layer in the second bloc of the encoder and decoder. It shouldn't be added inside the first blocs, the model doesnt have enougn time ot understand context and the gating of memory would be bad. \n"
  },
  {
    type: 'code',
    content: "import copy\nimport time\nimport math\nimport matplotlib.pyplot as plt\nimport pandas as pd\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\nfrom torch.optim.lr_scheduler import LambdaLR\nfrom torch.utils.data import DataLoader"
  },
  {
    type: 'code',
    content: "DATA_PATH = \"kaggle/input/wmt-2014-english-french/wmt14_translate_fr-en_train.csv\"\nDEVICE = \"mps\"\n\nCONFIG = {\n    \"batch_size\": 16,\n    \"feature_amount\": 512, # how precise our tokens are\n    \"vocab_size\": 32000,  # amount of distinct chunks of words\n    \"dataset_lenght\": 10000,  # lines of the CSV dataset\n    \"head_amount\": 8, # each head is a question/answer space\n    \"stream_amount\": 4, # amount of evil twins for each token\n    \"num_epochs\": 3,\n    \"accum_iter\": 10,  # amount of batches computed to increment the weights (effective batch = iter*batch_size)\n    \"base_lr\": 1.0,  # learning rate\n    \"max_padding\": 72, # length of sentences, filled with paddings to reach it\n    \"warmup\": 3000,  # linear increase of the learning rate\n    \"dropout_rate\": 0.1\n}"
  },
  {
    type: 'image',
    src: notebookImages["attention.png"],
    alt: "attention.png"
  },
  {
    type: 'code',
    content: "class EncoderDecoder(nn.Module):\n    \"the whole architecture.\"\n\n    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):\n        super().__init__()\n        self.encoder = encoder\n        self.decoder = decoder\n        self.src_embed = src_embed # first layer that turns a sentence into tokens + positions encoding\n        self.tgt_embed = tgt_embed\n        self.generator = generator # final layer that turns decoder outputs into probabilities\n\n    def forward(self, src, tgt, src_mask, tgt_mask):\n        # src: (B, L), tgt: (B, L-1)\n        # src_mask: (B, 1, L), to hide <pad>\n        # tgt_mask: (B, 1, L-1, L-1), to hide <pad> and future tokens\n        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)\n\n    def encode(self, src, src_mask):\n        x = self.src_embed(src) # x: (B, L, D)\n        B, L, D = x.shape\n        S = self.encoder.stream_amount\n        x = x.unsqueeze(2).expand(B, L, S, D) # x: (B, L, S, D)\n        return self.encoder(x, src_mask) # mask is meant to hide the <pad>\n\n    def decode(self, representation, src_mask, tgt, tgt_mask):\n        # representation: (B, L, S*D)\n        B, L, SD = representation.shape\n        S = self.decoder.stream_amount\n        \n        representation = representation.view(B, L, S, SD // S) # representation: (B, L, S, D)\n        representation = representation.mean(dim=2) # average over streams \n        representation = representation.unsqueeze(2) # representation: (B, L, 1, D)\n\n        x = self.tgt_embed(tgt)\n        x = x.unsqueeze(2).expand(B, x.size(1), S, SD // S) # x: (B, L, S, D)\n        return self.decoder(x, representation, src_mask, tgt_mask)"
  },
  {
    type: 'code',
    content: "def xavier_init(module):\n    \"Helper to initialize weights with Xavier uniform, ignoring biaises and scalars.\"\n    for p in module.parameters():\n        if p.dim() > 1:\n            nn.init.xavier_uniform_(p)"
  },
  {
    type: 'code',
    content: "def clones(module, N):\n    \"Helper to produce N identical layers.\"\n    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])"
  },
  {
    type: 'code',
    content: "class Generator(nn.Module):\n    \"Final layer that turns the logits into probabilities. Predict the next word.\"\n\n    def __init__(self, feature_amount, vocab):\n        super().__init__()\n        self.proj = nn.Linear(feature_amount, vocab)\n        xavier_init(self)\n\n    def forward(self, x):\n        # x: (B, L, D)\n        # output: (B, L, V)\n        probability = F.log_softmax(self.proj(x), dim=-1)\n        return probability"
  },
  {
    type: 'code',
    content: "class RMSNorm(nn.Module):\n    \"Only scale by root mean square. Computationaly more efficient. Doesn't hurt performance.\"\n\n    def __init__(self, river_size, eps=1e-6):\n        super().__init__()\n        self.scale = nn.Parameter(torch.ones(river_size))\n        self.eps = eps\n\n    def forward(self, x):\n        # x: (B, L, S*D)\n        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)\n        return self.scale * (x / rms) "
  },
  {
    type: 'code',
    content: "def doubly_stochastic(matrix, iterations=5): # Sinkhorn-Knopp algorithm for mHC\n    M = torch.exp(matrix)  # ensures positivity\n    for i in range(iterations): # alternate row and column normalization\n        M = M / M.sum(dim=-1, keepdim=True)\n        M = M / M.sum(dim=-2, keepdim=True)   \n    return M  "
  },
  {
    type: 'code',
    content: "class mHConnection(nn.Module):\n    \"\"\"\n    The newest Deepseek implementation of the mHC, replacing residual connection.\n    Implement three matrices, mixing streams after every layer.\n    \"\"\"\n\n    def __init__(self, stream_amount, feature_amount, dropout_rate=0.1):\n        super().__init__()\n        river_size = stream_amount*feature_amount\n        self.river_size = river_size\n        self.norm = RMSNorm(river_size)\n        self.dropout = nn.Dropout(dropout_rate) \n        \n        self.alpha_pre = nn.Parameter(torch.tensor(0.01))\n        self.alpha_post = nn.Parameter(torch.tensor(0.01))\n        self.alpha_res = nn.Parameter(torch.tensor(0.01))\n        \n        self.phi_pre = nn.Parameter(torch.empty(river_size, stream_amount))\n        self.phi_post = nn.Parameter(torch.empty(river_size, stream_amount))\n        self.phi_res = nn.Parameter(torch.empty(river_size, stream_amount, stream_amount))\n        xavier_init(self)\n\n        self.biais_pre = nn.Parameter(torch.zeros(stream_amount))\n        self.biais_post = nn.Parameter(torch.zeros(stream_amount))\n        self.biais_res = nn.Parameter(torch.eye(stream_amount) * 5) # bias: (S, S)\n        # 5 is the default value in the Deepseek paper\n        \n    def forward(self, x, sublayer):\n        # x: (B, L, S, D)\n        B, L, S, D = x.shape\n        x_river = x.flatten(start_dim=2) # shape: (B, L, S*D)\n        x_river = self.norm(x_river) # normalize over all streams of the token\n\n        H_pre = self.alpha_pre * x_river@self.phi_pre + self.biais_pre\n        H_post = self.alpha_post * x_river@self.phi_post + self.biais_post\n        H_res = self.alpha_res * torch.einsum('bld,dst->blst', x_river, self.phi_res) + self.biais_res\n\n        H_pre = F.sigmoid(H_pre).unsqueeze(2) # shape: (B, L, 1, S)\n        H_post = 2*F.sigmoid(H_post).unsqueeze(3) # shape: (B, L, S, 1)\n        H_res = doubly_stochastic(H_res) # shape: (B, L, S, S)\n        \n        # sublayer: (B, L, 1, D) -> (B, L, 1, D)\n        x_norm = x_river.view(B, L, S, D)\n        return H_res@x + H_post@self.dropout(sublayer(H_pre@x_norm))"
  },
  {
    type: 'code',
    content: "class Encoder(nn.Module):\n    \"\"\"\n    Encoder is a stack of N blocks, each block containing an engram layer (every two blocks), a\n    self-attention layer and a feed-forward layer, connected by mHconnections. \n    \"\"\"\n\n    def __init__(self, block, N, stream_amount=CONFIG[\"stream_amount\"]):\n        super().__init__()\n        river_size = block.feature_amount*stream_amount\n        self.blocks = clones(block, N)\n        self.norm = RMSNorm(river_size)\n        self.stream_amount = stream_amount\n\n    def forward(self, x, mask):\n        \"Pass the input through each block sequentially.\"\n        # x: (B, L, S, D)\n        # mask: (B, 1, L)\n        for block in self.blocks:\n            x = block(x, mask)\n        return self.norm(x.flatten(start_dim=2).contiguous()) # x: (B, L, S*D)"
  },
  {
    type: 'code',
    content: "class EncoderBlock(nn.Module):\n    \"An encoder block is made up of engram + self-attention + feed forward.\"\n    def __init__(self, feature_amount, stream_amount, engram, self_attn, feed_forward, dropout_rate, N):\n        super().__init__()\n        self.N = N\n        self.engram = engram\n        self.self_attn = self_attn\n        self.feed_forward = feed_forward\n        self.sublayers = clones(mHConnection(stream_amount, feature_amount, dropout_rate), 3)\n        self.feature_amount = feature_amount\n\n    def forward(self, x, mask):\n        # x: (B, L, S, D)\n        # mask: (B, 1, L)\n        if self.N % 2 == 0: # only an engram layer on the 2nd, 4th etc blocks\n            x = self.sublayers[0](x, lambda x: self.engram(x, mask))  # engram\n        x = self.sublayers[1](x, lambda x: self.self_attn(x, x, x, mask)) # attention\n        x = self.sublayers[2](x, self.feed_forward)  # feed forward\n        return x"
  },
  {
    type: 'code',
    content: "class Decoder(nn.Module):\n    \"Generic N blocks decoder with masking.\"\n\n    def __init__(self, block, N):\n        super().__init__()\n        stream_amount = CONFIG[\"stream_amount\"]\n        river_size = stream_amount*block.feature_amount\n\n        self.blocks = clones(block, N)\n        self.norm = RMSNorm(river_size)\n        self.stream_amount = stream_amount\n\n    def forward(self, x, memory, src_mask, tgt_mask):\n        # x: (B, L, S, D)\n        # memory: (B, L, S, D)\n        # src_mask: (B, 1, L)\n        # tgt_mask: (B, 1, L, L)\n        for block in self.blocks:\n            x = block(x, memory, src_mask, tgt_mask)\n        \n        x = self.norm(x.flatten(start_dim=2).contiguous()) # x: (B, L, S*D)\n        B, L, SD = x.shape\n        S = self.stream_amount\n        x = x.view(B, L, S, SD // S).mean(dim=2) # average the streams after last block\n        return x  # x: (B, L, D)"
  },
  {
    type: 'code',
    content: "class DecoderBlock(nn.Module):\n    \"A decoder block is made of 4 layers: engram, self-attention, source-attention, and feed forward.\"\n    def __init__(self, feature_amount, stream_amount, engram, self_attn, src_attn, feed_forward, dropout_rate, N):\n        super().__init__()\n        self.N = N\n        self.self_attn = self_attn\n        self.src_attn = src_attn\n        self.feed_forward = feed_forward\n        self.engram = engram\n        self.sublayers = clones(mHConnection(stream_amount, feature_amount, dropout_rate), 4)\n        self.feature_amount = feature_amount\n\n    def forward(self, x, memory, src_mask, tgt_mask):\n        # x: (B, L, S, D)\n        # memory: (B, L_src, S, D)\n        # src_mask: (B, 1, L_src)\n        # tgt_mask: (B, 1, L, L)\n        # output: (B, L, S, D)\n        m = memory\n        if self.N%2 == 0:\n            x = self.sublayers[0](x, lambda x: self.engram(x, tgt_mask))\n        x = self.sublayers[1](x, lambda x: self.self_attn(x, x, x, tgt_mask))\n        x = self.sublayers[2](x, lambda x: self.src_attn(x, m, m, src_mask))\n        x = self.sublayers[3](x, self.feed_forward)\n        return x"
  },
  {
    type: 'code',
    content: "def subsequent_mask(sentence_length):\n    \"Mask out subsequent positions.\"\n    attn_shape = (1, sentence_length, sentence_length)\n    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)\n    # zeros everything under the first line above the diagonal of the matrix\n    mask = (subsequent_mask == 0) # True/False instead of 0/1\n    return mask"
  },
  {
    type: 'code',
    content: "def attention(query, key, value, mask=None, dropout=None):\n    \"Scaled dot product between query and key, and distribute the scores on value.\"\n\n    d_k = query.size(-1)  # query/key smaller head_dimension (divider of feature_amount)\n    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)\n    # divide by sqrt(d_k) to not reach softmax plateau where gradient is almost zero\n\n    if mask is not None:\n        scores = scores.masked_fill(mask == 0, -1e9)\n        \n    p_attn = scores.softmax(dim=-1) # probability of key answering query\n    if dropout is not None:\n        p_attn = dropout(p_attn)\n\n    return torch.matmul(p_attn, value), p_attn"
  },
  {
    type: 'code',
    content: "def frequencies(feature_amount, max_padding, theta = 1000):\n    \"\"\"Precompute the frequencies for all token positions in the sentence and\n    for all pairs of features.\"\"\"\n    freqs = 1.0 / (theta ** (torch.arange(0, feature_amount, 2) / feature_amount))\n    # the first feature pairs  have high frequency, hence distance sensitive, the final pairs have low frequencies and are capture global properties\n    t = torch.arange(max_padding)\n    freqs = torch.outer(t, freqs).float() # all possible products\n    return torch.cos(freqs), torch.sin(freqs)\n\ndef embed_positions(A, cos, sin):\n    \"\"\"Apply the rotation to q and k to embed token relative positions.\"\"\"\n    cos, sin = (x[:A.size(-2)].to(A)[None, None].repeat_interleave(2, dim=-1) for x in (cos, sin))\n    x1, x2 = A.chunk(2, dim=-1)\n    return A * cos + torch.cat((-x2, x1), dim=-1) * sin"
  },
  {
    type: 'code',
    content: "class MultiHeadedAttention(nn.Module):\n    \"\"\"\n    Multi-head attention module: projects into smaller dimension spaces \n    and then apply specialized attention in each of them.\n    \"\"\"\n\n    def __init__(self, head_amount, feature_amount, max_padding, dropout_rate):\n        super().__init__()\n        self.head_dimension = feature_amount // head_amount # feature_amount = head_amount * head_dimension\n        self.head_amount = head_amount\n        self.linears = clones(nn.Linear(feature_amount, feature_amount, bias=False), 4) # Q, K, V, Out \n        self.attn = None\n        self.dropout = nn.Dropout(dropout_rate)\n\n        cos, sin = frequencies(self.head_dimension, max_padding)\n        self.register_buffer(\"sin\", sin) # to move them on DEVICE with the instance\n        self.register_buffer(\"cos\", cos)\n        xavier_init(self)\n\n    def forward(self, query, key, value, mask=None):\n        # query, key, value: (B, L, 1, D)\n        # mask: (B, 1, L)\n        B, L, _, D = query.size()\n\n        if mask is not None:\n            mask = mask.unsqueeze(1) # mask: (B, 1, 1, L)\n            \n        # MPS FIX: Ensure 3D input for Linear layers (B, L, D) instead of (B, L, 1, D)\n        query = query.view(B, -1, D)\n        key = key.view(B, -1, D)\n        value = value.view(B, -1, D)\n\n        query, key, value = [\n            lin(x).view(B, -1, self.head_amount, self.head_dimension).transpose(1, 2)\n            for lin, x in zip(self.linears, (query, key, value))\n        ] # q, k, v: (B, H, L, D/H)\n        # we use -1 instead of L because it can be either L or L-1\n\n        query = embed_positions(query, self.cos, self.sin)\n        key = embed_positions(key, self.cos, self.sin)\n        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)\n        \n        # MPS FIX: Flatten to 3D for final linear layer (B, L, D)\n        x = x.transpose(1, 2).reshape(B, L, self.head_amount * self.head_dimension)\n        \n        return self.linears[3](x).unsqueeze(2) # apply final linear projection and restore 4D"
  },
  {
    type: 'code',
    content: "class PositionwiseFeedForward(nn.Module):\n    \"Implements FFN bloc, composed of 2 fully connected layers\"\n\n    def __init__(self, feature_amount, d_ff, dropout_rate):\n        super().__init__()\n        self.w_1 = nn.Linear(feature_amount, d_ff) # d_ff is usually 4 times feature_amount\n        self.w_2 = nn.Linear(d_ff, feature_amount)\n        self.dropout = nn.Dropout(dropout_rate)\n        xavier_init(self)\n\n    def forward(self, x):\n        # x: (B, L, 1, D)\n        x = self.w_1(x)\n        x = x.relu()\n        x = self.dropout(x)\n        x = self.w_2(x)\n        return x"
  },
  {
    type: 'code',
    content: "class SimpleLossCompute:\n    \"Compute loss.\"\n\n    def __init__(self, generator, criterion):\n        self.generator = generator \n        self.criterion = criterion # the loss\n\n    def __call__(self, x, y, norm):\n        # norm is amounf of tokens (excluding paddings)\n        x = self.generator(x)\n        loss = (\n            self.criterion(x.reshape(-1, x.size(-1)), y.reshape(-1))\n            / norm  # average loss per token\n        )\n        return loss.data * norm, loss \n        # we return the unscaled loss too to reaverage over an epoch later on"
  },
  {
    type: 'code',
    content: "def make_model(src_vocab, tgt_vocab, stream_amount=CONFIG[\"stream_amount\"], \n               N=2, feature_amount=CONFIG[\"feature_amount\"], d_ff=2048, h=CONFIG[\"head_amount\"], \n               max_padding=CONFIG[\"max_padding\"], dropout_rate=CONFIG[\"dropout_rate\"]):\n    \"Construct an instance of the model from hyperparameters.\"\n\n    river_size = feature_amount*stream_amount\n    c = copy.deepcopy # makes an instance out of an instance, basically \"classifying\" an instance\n    attn = MultiHeadedAttention(h, feature_amount, max_padding, dropout_rate)\n    ff = PositionwiseFeedForward(feature_amount, d_ff, dropout_rate)\n    engram = Engram(feature_amount)\n\n    model = EncoderDecoder(\n        Encoder(EncoderBlock(feature_amount, stream_amount, c(engram), c(attn), c(ff), dropout_rate, N), N),\n        Decoder(DecoderBlock(feature_amount, stream_amount, c(engram), c(attn), c(attn), c(ff), dropout_rate, N), N),\n        nn.Sequential(Embeddings(feature_amount, src_vocab)),\n        nn.Sequential(Embeddings(feature_amount, tgt_vocab)),\n        Generator(feature_amount, tgt_vocab),\n    )\n\n    return model"
  },
  {
    type: 'code',
    content: "class Batch:\n    \"Object for holding a batch of data with mask during training.\"\n    def __init__(self, src, tgt=None, pad=2):  # 2 = <blank>\n        self.src = src\n        self.src_mask = (src != pad).unsqueeze(-2)\n        if tgt is not None:\n            self.tgt = tgt[:, :-1]\n            self.tgt_y = tgt[:, 1:]\n            self.tgt_mask = self.make_std_mask(self.tgt, pad)\n            self.ntokens = (self.tgt_y != pad).data.sum()\n\n    def make_std_mask(self, tgt, pad):\n        \"Create a mask to hide padding and future words.\"\n        tgt_mask = (tgt != pad).unsqueeze(-2)\n        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)\n        return tgt_mask"
  },
  {
    type: 'code',
    content: "def rate(step, model_size, factor, warmup):\n    \"Adaptive step rate, linear warmup, then square root decay\"\n    if step == 0:\n        step = 1\n    rate = factor * (model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5)))\n    return rate"
  },
  {
    type: 'code',
    content: "class LabelSmoothing(nn.Module):\n    \"\"\"\n    Implement label smoothing. Prevent trying to reach P = 1 or 0 \n    and overfit. Increase perplexity, but improve BLEU score.\n    \"\"\"\n\n    def __init__(self, feature_amount, padding_idx, smoothing=0.0):\n        super().__init__()\n        self.criterion = nn.KLDivLoss(reduction=\"sum\")\n        self.padding_idx = padding_idx\n        self.confidence = 1.0 - smoothing\n        self.smoothing = smoothing\n        self.feature_amount = feature_amount\n        self.true_dist = None\n\n    def forward(self, x, target):\n        assert x.size(1) == self.feature_amount\n        true_dist = x.data.clone()\n        true_dist.fill_(self.smoothing / (self.feature_amount - 2))\n        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)\n        true_dist[:, self.padding_idx] = 0\n        mask = torch.nonzero(target.data == self.padding_idx)\n        if mask.dim() > 0:\n            true_dist.index_fill_(0, mask.squeeze(), 0.0)\n        self.true_dist = true_dist\n        return self.criterion(x, true_dist.clone().detach())"
  },
  {
    type: 'code',
    content: "def XORhash(head_number, engram):\n    \"k-th head deterministic multiplicative-XOR hash.\"\n    if engram.dim() == 0:\n        return ((engram * (6364136223846793005 + head_number)) ^ 1469598103934665603) & 0x7FFFFFFFFFFFFFFF\n    if engram.dim() == 1:\n        engram = engram.unsqueeze(-1)\n    h = torch.full(engram.shape[:-1], 1469598103934665603 + head_number, dtype=torch.int64, device=engram.device)\n    for i in range(engram.size(-1)):\n        h = (h * 6364136223846793005) ^ (engram[..., i] + i + 1)\n    return h & 0x7FFFFFFFFFFFFFFF"
  },
  {
    type: 'code',
    content: "def embedding_sizes(engram_max_lenght, hash_head_amount):\n    \"One unique prime size per n-gram order and hash head number, increasing as ^n.\"\n    def next_prime(n):\n        while any(n % d == 0 for d in range(2, int(n**0.5) + 1)):\n            n += 1\n        return n\n\n    sizes, used = [], set()\n    for n in range(engram_max_lenght):\n        base = (20 + 2 ** n) // hash_head_amount\n        for k in range(hash_head_amount):\n            p = next_prime(base + k)\n            while p in used:\n                p = next_prime(p + 1)\n            sizes.append(p)\n            used.add(p)\n    return sizes"
  },
  {
    type: 'code',
    content: "class EngramConv(nn.Module):\n    \"Final layer of the engram layer, a convolution.\"\n    def __init__(self, feature_amount, kernel_size, dilation):\n        super().__init__()\n        self.norm = nn.RMSNorm(feature_amount)\n        self.dw_conv = nn.Conv1d(\n            in_channels=feature_amount,\n            out_channels=feature_amount,\n            kernel_size=kernel_size,\n            dilation=dilation,\n            groups=feature_amount,   # depthwise\n            padding=0\n        )\n        self.left_pad = (kernel_size - 1) * dilation\n\n    def forward(self, x):          \n        x = self.norm(x).transpose(1, 2)  # x: (B, D, L)       \n        x = F.pad(x, (self.left_pad, 0))      # causal pad\n        x = self.dw_conv(x).transpose(1, 2)  \n        return F.silu(x) + x # residual connection"
  },
  {
    type: 'code',
    content: "class Engram(nn.Module):\n    \"\"\"\n    Engram layer, with k-head hashing of the n-lenght engram at position t.\n    This stores memory of the model, and offset other layers to focus on reasoning.\n    \"\"\"\n    def __init__(self, feature_amount, hash_head_amount=4, engram_max_lenght=4):\n        super().__init__()\n        self.prime_sizes = embedding_sizes(engram_max_lenght, hash_head_amount)\n        self.hash_head_amount = hash_head_amount\n        self.engram_max_lenght = engram_max_lenght\n\n        self.tables = nn.ModuleList([\n            nn.Embedding(size, feature_amount) for size in self.prime_sizes\n        ]) # lists of the embedding tables tensors, prime around of slots, return a vector of lenght feature_amount\n        memory_lenght = feature_amount*hash_head_amount*(engram_max_lenght-1) # lenght of the memory vector\n        self.key = nn.Linear(memory_lenght, feature_amount, bias=False)\n        self.value = nn.Linear(memory_lenght, feature_amount, bias=False)\n        self.convolution = EngramConv(feature_amount, kernel_size=4, dilation=engram_max_lenght)\n\n        self.norm = RMSNorm(feature_amount)\n        self.feature_amount = feature_amount\n        xavier_init(self)\n        \n    def forward(self, x, mask):\n        # x: (B, L, 1, D)\n        B, L, _, D = x.shape\n        x = x.squeeze(2)\n        memory = torch.zeros_like(x)\n\n        table_idx, parts = 0, []\n        for n in range(2, self.engram_max_lenght + 1):\n            # context_n: (B, L, n, D), n tokens preceding position t\n            context_n = torch.stack([\n                F.pad(x[:, :-i, :], (0, 0, i, 0)) for i in range(1, n + 1)\n            ], dim=2)\n            g = torch.round(context_n.reshape(B, L, -1) * 1000).to(torch.int64)\n\n            for k in range(self.hash_head_amount):\n                z = XORhash(k, g) % self.prime_sizes[table_idx]\n                parts.append(self.tables[table_idx](z)) \n                table_idx += 1\n\n        memory = torch.cat(parts, dim=-1)  # (B, L, memory_lenght)\n        alpha = (self.norm(x)*self.norm(self.key(memory))).sum(dim=2) # if context align to memory, gates fully opened\n        alpha = torch.sigmoid(alpha/math.sqrt(self.feature_amount)) # like attention, we divide by sqrt(D) to get unit variance\n        gated_memory = alpha.unsqueeze(2)*self.value(memory) # gated_memory: (B, L, D)\n        out = self.convolution(gated_memory)\n        return out.unsqueeze(2)"
  },
  {
    type: 'image',
    src: notebookImages["engram-2.png"],
    alt: "engram-2.png"
  },
  {
    type: 'markdown',
    content: "# Tokenizing and embedding"
  },
  {
    type: 'code',
    content: "from tokenizers import Tokenizer\nfrom tokenizers.models import BPE\nfrom tokenizers.trainers import BpeTrainer\nfrom tokenizers.pre_tokenizers import Metaspace\nfrom tokenizers.processors import TemplateProcessing\nfrom tokenizers.decoders import Metaspace as MetaspaceDecoder"
  },
  {
    type: 'code',
    content: "def load_data(data_path=DATA_PATH):\n    df = pd.read_csv(\n        data_path, \n        engine='python',        # Robust parser\n        on_bad_lines='skip',\n        nrows=CONFIG['dataset_lenght']\n    )    \n    \n    def clean(text):\n        if not isinstance(text, str): return text\n        replacements = {\n            \"Ã©\": \"é\", \n            \"Ã¨\": \"è\", \n            \"Ã \": \"à\", \n            \"Ã \": \"à\", \n            \"Ã´\": \"ô\", \n            \"Ã»\": \"û\", \n            \"Ã¯\": \"ï\",\n            \"Ã¹\": \"ù\", \n            \"Ãî\": \"î\", \n            \"Ã‰\": \"É\", \n            \"Â \": \" \",      \n            \"â–\": \"_\",  \n            \"Â»\": \"»\", \n            \"Â«\": \"«\",  \n            \"Ã§\": \"ç\",\n            \"Ã®\": \"î\",\n            \"Ãª\": \"ê\",\n            \"Ã€\": \"À\",\n            \"Ã‡\": \"Ç\"\n        }\n        for bad, good in replacements.items():\n            text = text.replace(bad, good)\n        return text\n\n    df['fr'] = df['fr'].apply(clean)\n    df['en'] = df['en'].apply(clean)\n    \n    train_src = df['fr'].tolist() # Source is French\n    train_tgt = df['en'].tolist() # Target is English\n    return train_src, train_tgt"
  },
  {
    type: 'code',
    content: "def train_tokenizer(data, vocab_size, min_frequency=2):\n    \"Construct a tokenizer using Byte-Pair Encoding\"\n\n    tokenizer = Tokenizer(BPE(unk_token=\"<unk>\"))\n    tokenizer.pre_tokenizer = Metaspace(replacement=\"_\") # use _ for spaces\n    trainer = BpeTrainer(special_tokens=[\"<s>\", \"<pad>\", \"</s>\", \"<unk>\"], \n                         vocab_size=vocab_size, \n                         min_frequency=min_frequency)\n    tokenizer.train_from_iterator(data, trainer)\n    \n    # post-processing: add <s> at start and end\n    tokenizer.post_processor = TemplateProcessing(\n        single=\"<s> $A </s>\",\n        pair=\"<s> $A </s> $B:1 </s>:1\",\n        special_tokens=[\n            (\"<s>\", tokenizer.token_to_id(\"<s>\")),\n            (\"</s>\", tokenizer.token_to_id(\"</s>\")),\n        ],\n    )\n    tokenizer.decoder = MetaspaceDecoder(replacement=\"_\")\n    return tokenizer"
  },
  {
    type: 'code',
    content: "def load_tokenizers(data_path=DATA_PATH):\n    \"Read the csv and returns two JSON containing the tokenizers\"\n    train_src, train_tgt = load_data(data_path=DATA_PATH)\n    tokenizer_src = train_tokenizer(train_src, vocab_size=CONFIG['vocab_size'])\n    tokenizer_tgt = train_tokenizer(train_tgt, vocab_size=CONFIG['vocab_size'])\n    tokenizer_src.save(\"tokenizers/tokenizer_src.json\")\n    tokenizer_tgt.save(\"tokenizers/tokenizer_tgt.json\")\n    return tokenizer_src, tokenizer_tgt"
  },
  {
    type: 'code',
    content: "class Embeddings(nn.Module):\n    def __init__(self, feature_amount, vocab):\n        super().__init__()\n        self.embedding_table = nn.Embedding(vocab, feature_amount)\n        self.feature_amount = feature_amount\n        xavier_init(self)\n\n    def forward(self, x):\n        # x: (B, L) (indices)\n        # output: (B, L, S, d)\n        return self.embedding_table(x) * math.sqrt(self.feature_amount)"
  },
  {
    type: 'markdown',
    content: "# Training"
  },
  {
    type: 'code',
    content: "def run_epoch(data_iter, model, loss_compute, optimizer=None, scheduler=None, mode=\"train\",\n    accum_iter=1):\n    \"Runs one epoch, either for training, either validation.\"\n    \n    start = time.time()\n    total_tokens, total_loss, tokens, n_accum = 0, 0, 0, 0\n    loss_list, grad_norms = [], []\n\n    for i, batch in enumerate(data_iter):\n        out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)\n        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)\n\n        if mode == \"train\":\n            (loss_node / accum_iter).backward()\n            if i % accum_iter == 0:\n                if optimizer is not None:\n                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))\n                    grad_norms.append(norm.item())\n                    optimizer.step()\n                    optimizer.zero_grad(set_to_none=True)\n                n_accum += 1\n            if scheduler is not None:\n                scheduler.step()\n\n        total_loss += loss\n        total_tokens += batch.ntokens\n        tokens += batch.ntokens\n        loss_list.append((loss / batch.ntokens).item())\n\n        if i % 40 == 1 and (mode == \"train\"): # print every 40 batches\n            lr = optimizer.param_groups[0][\"lr\"]\n            elapsed = time.time() - start\n            print(\n                (\"Epoch: %6d | Accum. Step: %3d | Loss: %6.2f \" + \"| Tokens/sec: %7.1f | Learn. Rate: %6.1e\")\n                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)\n            )\n            start = time.time()\n            tokens = 0\n\n    return loss_list, grad_norms"
  },
  {
    type: 'code',
    content: "def collate_batch(batch, src_tokenizer, tgt_tokenizer, max_padding=CONFIG[\"max_padding\"], pad_id=1):\n    \"Makes every sentence have the same lenght of max_padding to get a tensor\"\n    src_list, tgt_list = [], []\n    \n    for (sentence_src, sentence_tgt) in batch:\n        encoded_src = src_tokenizer.encode(sentence_src).ids\n        encoded_tgt = tgt_tokenizer.encode(sentence_tgt).ids\n        src_tensor = torch.tensor(encoded_src, dtype=torch.int64, device=DEVICE)\n        tgt_tensor = torch.tensor(encoded_tgt, dtype=torch.int64, device=DEVICE)\n\n        # pad to reach max length\n        src_list.append(F.pad(src_tensor, (0, max_padding - len(src_tensor)), value=pad_id))\n        tgt_list.append(F.pad(tgt_tensor, (0, max_padding - len(tgt_tensor)), value=pad_id))\n\n    src = torch.stack(src_list)\n    tgt = torch.stack(tgt_list)\n    return (src, tgt)"
  },
  {
    type: 'code',
    content: "def create_dataloaders(vocab_src, vocab_tgt, batch_size=12000, max_padding=CONFIG[\"max_padding\"]):\n    \"Instantiate the dataloaders for training and validation.\"\n  \n    pad_id = vocab_src.token_to_id(\"<pad>\") # get pad ID dynamically\n    def collate_fn(batch):\n        return collate_batch(\n            batch,\n            vocab_src,\n            vocab_tgt,\n            max_padding=max_padding,\n            pad_id=pad_id,\n        )\n\n    train_src, train_tgt = load_data(DATA_PATH)\n    split_idx = int(len(train_src) * 0.95)\n    train_iter = list(zip(train_src[:split_idx], train_tgt[:split_idx]))\n    valid_iter = list(zip(train_src[split_idx:], train_tgt[split_idx:]))\n\n    train_dataloader = DataLoader(\n        train_iter,\n        batch_size=batch_size,\n        shuffle=True,\n        collate_fn=collate_fn,\n    )\n    valid_dataloader = DataLoader(\n        valid_iter,\n        batch_size=batch_size,\n        shuffle=False,\n        collate_fn=collate_fn,\n    )\n    return train_dataloader, valid_dataloader"
  },
  {
    type: 'code',
    content: "def train_model(tokenizer_src, tokenizer_tgt):\n    \" Launch the training process of a model\"\n    print(\"Training process starting...\", flush=True)\n    \n    pad_idx = tokenizer_tgt.token_to_id(\"<pad>\")\n    model = make_model(tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size())\n    device = torch.device(DEVICE)\n    model.to(device)\n    model_param_count = sum(p.numel() for p in model.parameters())\n    \n    criterion = LabelSmoothing(feature_amount=tokenizer_tgt.get_vocab_size(), padding_idx=pad_idx, smoothing=0.1)\n    criterion.to(device)\n\n    train_dataloader, valid_dataloader = create_dataloaders(\n        tokenizer_src,\n        tokenizer_tgt,\n        batch_size=CONFIG[\"batch_size\"],\n        max_padding=CONFIG[\"max_padding\"])\n\n    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG[\"base_lr\"], betas=(0.9, 0.98), eps=1e-9)\n    lr_scheduler = LambdaLR(\n        optimizer,\n        lr_lambda=lambda step: rate(\n            step, CONFIG[\"feature_amount\"], factor=1, warmup=CONFIG[\"warmup\"]\n        ) # personalized learning rate\n    )\n\n    train_losses, valid_losses, all_grad_norms = [], [], []\n    for epoch in range(CONFIG[\"num_epochs\"]):\n        model.train()\n        print(f\"Epoch n°{epoch} Training ====\", flush=True)\n        loss_list, grad_norms = run_epoch(\n            (Batch(b[0], b[1], pad_idx) for b in train_dataloader),\n            model,\n            SimpleLossCompute(model.generator, criterion),\n            optimizer,\n            lr_scheduler,\n            mode=\"train\",\n            accum_iter=CONFIG[\"accum_iter\"],\n        )\n        train_losses.extend(loss_list)\n        all_grad_norms.extend(grad_norms)\n\n        file_path = \"checkpoints/%s%.2d.pt\" % (\"checkpoint_\", epoch)\n        torch.save(model.state_dict(), file_path)\n\n        print(f\"Epoch n°{epoch} Validation ====\", flush=True)\n        model.eval()\n        loss_list, _ = run_epoch(\n            (Batch(b[0], b[1], pad_idx) for b in valid_dataloader),\n            model,\n            SimpleLossCompute(model.generator, criterion),\n            None,\n            None,\n            mode=\"eval\",\n        )\n        valid_losses.extend(loss_list)\n\n        mean_val = sum(loss_list) / len(loss_list)\n        print(mean_val)    \n            \n    plot(train_losses, valid_losses, all_grad_norms, model_param_count)"
  },
  {
    type: 'code',
    content: "def plot(train_losses, valid_losses, all_grad_norms, model_param_count):\n    \"Plot the losses and the gradient norm over the training process.\"\n    num_val_batches = len(valid_losses) // CONFIG[\"num_epochs\"]\n    epoch_mean_valid_losses, epoch_means = [], []\n    for i in range(CONFIG[\"num_epochs\"]):\n        epoch_data = valid_losses[i*num_val_batches : (i+1)*num_val_batches]\n        mean_val = sum(epoch_data) / len(epoch_data)\n        epoch_mean_valid_losses.extend([mean_val] * len(epoch_data))\n        epoch_means.append(mean_val)\n\n    plt.figure(figsize=(10, 10))\n    \n    ax1 = plt.subplot(2, 1, 1) # subplot 1: loss\n    train_batches_per_epoch = len(train_losses) / CONFIG[\"num_epochs\"]\n    x_train = [i / train_batches_per_epoch for i in range(len(train_losses))]\n    valid_batches_per_epoch = len(valid_losses) / CONFIG[\"num_epochs\"]\n    x_valid = [i / valid_batches_per_epoch for i in range(len(valid_losses))]\n    window_size = max(1, int(train_batches_per_epoch * 0.1))\n    train_loss_rolling = pd.Series(train_losses).rolling(window=window_size, center=True).mean()\n    plt.plot(x_train, train_losses, label='Training Loss')\n    plt.plot(x_train, train_loss_rolling, label='Training Loss (Avg)', color='darkblue', linewidth=1)\n    plt.plot(x_valid, valid_losses, label='Validation Loss', alpha=0.7)\n    plt.plot(x_valid, epoch_mean_valid_losses, label='Val Loss (Avg)', color='red', linewidth=1, linestyle='--')\n    for i, mean_val in enumerate(epoch_means):\n        plt.text(i + 0.5, mean_val, f\"{mean_val:.3g}\", \n                 color='red', \n                 ha='center', \n                 va='bottom', \n                 fontweight='medium')\n    plt.xlabel('Epochs')\n    plt.ylabel('Loss')\n    plt.legend()\n    plt.title('Training and Validation Loss Over Epochs')\n    ax1.set_xlim(0, CONFIG[\"num_epochs\"])\n    ax1.set_xticks(range(CONFIG[\"num_epochs\"] + 1))\n        \n    plt.subplot(2, 1, 2) # subplot 2: gradient norm\n    plt.plot(all_grad_norms, label='Gradient Norm', color='orange')\n    plt.xlabel('Steps')\n    plt.ylabel('Norm')\n    plt.legend()\n    plt.title('Gradient Norm over Steps')\n    ax = plt.gca()\n    ax.set_xticks(range(len(all_grad_norms)))\n    ax.text(0.99, 0.02, f'Params count: {model_param_count:,}', transform=ax.transAxes,\n            ha='right', va='bottom', fontsize=9, color='dimgray')\n\n    plt.tight_layout()\n    plt.show()"
  },
  {
    type: 'code',
    content: "# launch the training process\nif __name__ == \"__main__\":\n    print(f\"Dataset found at {DATA_PATH} Initializing training...\")\n    tokenizer_src, tokenizer_tgt = load_tokenizers(data_path=DATA_PATH) \n    train_model(tokenizer_src, tokenizer_tgt)"
  },
  {
    type: 'markdown',
    content: "# Inference"
  },
  {
    type: 'code',
    content: "from torchmetrics.text import BLEUScore\nfrom tqdm import tqdm"
  },
  {
    type: 'code',
    content: "def greedy_decode(model, src, src_mask, max_len, start_symbol):\n    memory = model.encode(src, src_mask)\n    # use the batch size from the source\n    batch_size = src.size(0)\n    ys = torch.zeros(batch_size, 1).fill_(start_symbol).type_as(src.data)\n    \n    for i in range(max_len - 1):\n        out = model.decode(\n            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)\n        )\n        prob = model.generator(out[:, -1])\n        _, next_word = torch.max(prob, dim=1)\n        # next_word is (batch_size,) -> unsqueeze to (batch_size, 1) for concat\n        ys = torch.cat(\n            [ys, next_word.unsqueeze(1)], dim=1\n        )\n    return ys"
  },
  {
    type: 'code',
    content: "def check_outputs(dataloader, model, tokenizer_src, tokenizer_tgt,\n    n_examples=5, pad_idx=1, eos_string=\"</s>\"):\n    \"Check the models outputs against the ground truth\"\n\n    results = [()] * n_examples\n    valid_iter = iter(dataloader) \n    \n    for idx in range(n_examples):\n        print(\"\\nExample %d ========\\n\" % idx)\n        try:\n            # Get the NEXT batch from the iterator\n            b = next(valid_iter)\n        except StopIteration:\n            print('No more examples in test set.')\n            break\n            \n        rb = Batch(b[0], b[1], pad_idx)\n        \n        src_text = tokenizer_src.decode(rb.src[0].tolist(), skip_special_tokens=True)\n        tgt_text = tokenizer_tgt.decode(rb.tgt[0].tolist(), skip_special_tokens=True)\n\n        print(\"Source Text (Input)        : \" + src_text)\n        print(\"Target Text (Ground Truth) : \" + tgt_text)\n        \n        model_out = greedy_decode(model, rb.src, rb.src_mask, 72, 0)[0]\n        model_txt = tokenizer_tgt.decode(model_out.tolist(), skip_special_tokens=True)\n        \n        print(\"Model Output               : \" + model_txt)\n        results[idx] = (rb, src_text, tgt_text, model_out, model_txt)\n        \n    return results"
  },
  {
    type: 'code',
    content: "def calculate_bleu(model, dataloader, tokenizer_tgt, device=DEVICE):\n    \"Compute BLEU score, which measures n-gram overlaps\"\n    model.eval()\n    metric = BLEUScore()\n    \n    preds, targets = [], []\n    pad_idx = tokenizer_tgt.token_to_id(\"<pad>\")\n    start_symbol = tokenizer_tgt.token_to_id(\"<s>\")\n    \n    with torch.no_grad():\n        for b in tqdm(dataloader, desc=\"Computing BLEU\"):\n            src = b[0].to(device)  # b[0]: src, b[1]: tgt \n            src_mask = (src != pad_idx).unsqueeze(-2)\n    \n            model_out = greedy_decode(model, src, src_mask, max_len=72, start_symbol=start_symbol)\n            preds.extend(tokenizer_tgt.decode_batch(model_out.cpu().tolist(), skip_special_tokens=True))     \n            # Decode Ground Truth (b[1])\n            ref_texts = tokenizer_tgt.decode_batch(b[1].tolist(), skip_special_tokens=True)\n            targets.extend([[t] for t in ref_texts])\n\n    score = metric(preds, targets)\n    print(f\"BLEU Score: {score.item():.4f}\")"
  },
  {
    type: 'code',
    content: "if __name__ == \"__main__\":\n    \n    tokenizer_src = Tokenizer.from_file(\"tokenizers/kaggle_tokenizer_src.json\")\n    tokenizer_tgt = Tokenizer.from_file(\"tokenizers/kaggle_tokenizer_tgt.json\")\n    \n    model = make_model(tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size(), stream_amount=4, N=6, feature_amount=512, d_ff=2048, h=8, dropout_rate=0.1)\n    checkpoint_path = \"checkpoints/checkpoint_02.pt\"\n    print(f\"Loading {checkpoint_path}...\")\n    \n    num_params = sum(p.numel() for p in model.parameters())\n    print(f\"Number of parameters: {num_params}\")\n\n    # Load the saved weights to the model\n    state_dict = torch.load(checkpoint_path, map_location=torch.device(DEVICE))\n    model.load_state_dict(state_dict)\n    model.to(DEVICE)\n    model.eval()\n\n    _, valid_dataloader = create_dataloaders(\n        tokenizer_src,\n        tokenizer_tgt,\n        batch_size=32)\n\n    check_outputs(valid_dataloader, model, tokenizer_src, tokenizer_tgt)\n    calculate_bleu(model, valid_dataloader, tokenizer_tgt)"
  },
  {
    type: 'markdown',
    content: "# Deprecated"
  },
  {
    type: 'code',
    content: "class LayerNorm(nn.Module):\n    \"Offset average and scale by variance the hidden states.\"\n\n    def __init__(self, feature_amount, eps=1e-6):\n        super().__init__()\n        self.scale = nn.Parameter(torch.ones(feature_amount))\n        self.offset = nn.Parameter(torch.zeros(feature_amount))\n        self.eps = eps  # to not divide by zero\n\n    def forward(self, x):\n        # x: (B, L, S*D)\n        mean = x.mean(-1, keepdim=True)\n        std = x.std(-1, keepdim=True)\n        return self.scale * (x - mean) / (std + self.eps) + self.offset"
  },
  {
    type: 'code',
    content: "class ResidualConnection(nn.Module):\n    \"A residual connection followed by a layer norm.\"\n\n    def __init__(self, feature_amount, dropout_rate):\n        super().__init__()\n        self.norm = LayerNorm(feature_amount)\n        self.dropout = nn.Dropout(dropout_rate) \n        # zeros randomly a part of the weights during training (not inference)\n\n    def forward(self, x, sublayer):\n        # x: (B, L, S, D)\n        # sublayer: (B, L, S, d) -> (B, L, S, d)\n        \"Apply residual connection to any sublayer with the same size.\"\n        return x + self.dropout(sublayer(self.norm(x)))"
  },
  {
    type: 'code',
    content: "class PositionalEncoding(nn.Module):\n    \"\"\"Implement the PE function, before we split tokens into streams. \n    We need this because of the dot product in the attention layers, which is permutation invariant.\"\"\"\n\n    def __init__(self, feature_amount, dropout_rate, max_len=5000):\n        super().__init__()\n        self.dropout = nn.Dropout(dropout_rate)\n\n        # compute the positional encodings once in log space.\n        pe = torch.zeros(max_len, feature_amount)\n        position = torch.arange(0, max_len).unsqueeze(1)\n        div_term = torch.exp(torch.arange(0, feature_amount, 2) * -(math.log(10000.0) / feature_amount))\n        pe[:, 0::2] = torch.sin(position * div_term)\n        pe[:, 1::2] = torch.cos(position * div_term)\n        pe = pe.unsqueeze(0)\n        self.register_buffer(\"pe\", pe)\n\n    def forward(self, x):\n        # x: (B, L, D)\n        # output: (B, L, D)\n        x = x + self.pe[:, :x.size(1)].requires_grad_(False)\n        x = self.dropout(x)\n        return x"
  }
]);

function renderMarkdown(text) {
  // 1. Temporarily replace markdown links to avoid double-linking
  const links = [];
  let html = text.replace(/\[(.*?)\]\((.*?)\)/gim, (match, label, url) => {
    links.push({ label, url });
    return `__MD_LINK_${links.length - 1}__`;
  });

  // 2. Auto-link raw URLs - use 'link' text
  html = html.replace(/(https?:\/\/[^\s]+)/gim, '<a href="$1" target="_blank" class="md-link">link</a>');
  html = html.replace(/__MD_LINK_(\d+)__/gim, (match, id) => {
    const link = links[parseInt(id)];
    return `<a href="${link.url}" target="_blank" class="md-link">${link.label}</a>`;
  });

  // 4. Standard markdown formatting
  html = html
    .replace(/^# (.*$)/gim, '<h1>$1</h1>')
    .replace(/^## (.*$)/gim, '<h2>$1</h2>')
    .replace(/^### (.*$)/gim, '<h3>$1</h3>')
    .replace(/\*\*(.*?)\*\*/gim, '<strong>$1</strong>')
    .replace(/\$\$(.*?)\$\$/gim, '<div class="math">$$$1$$</div>') 
    .replace(/\n/gim, '<br />');
  
  return html;
}

function getHighlightedLines(code) {
  if (!code) return [];
  const highlighted = highlight(code);
  return splitHtmlByLines(highlighted);
}

function splitHtmlByLines(html) {
  const lines = [];
  let currentLine = '';
  // Stack of open tags to ensure we close/re-open them across lines
  const tagStack = [];
  
  // Regex to match:
  // 1. Opening span tags: <span class="...">
  // 2. Closing span tags: </span>
  // 3. Text content (non-tag characters)
  const regex = /(<span [^>]+>)|(<\/span>)|([^<]+)/g;
  
  let match;
  while ((match = regex.exec(html)) !== null) {
    const [full, openTag, closeTag, text] = match;
    
    if (openTag) {
      currentLine += openTag;
      tagStack.push(openTag);
    } else if (closeTag) {
      currentLine += closeTag;
      tagStack.pop();
    } else if (text) {
      const parts = text.split('\n');
      for (let i = 0; i < parts.length; i++) {
        const part = parts[i];
        currentLine += part;
        
        // If there are more parts, it means we hit a newline
        if (i < parts.length - 1) {
            // Close all currently open tags for the current line
            // We need to close them in reverse order (LIFO)
            // Since all our tags are </span>, we just append it
            for (let j = 0; j < tagStack.length; j++) {
                currentLine += '</span>';
            }
            lines.push(currentLine);
            currentLine = '';
            
            // Re-open all tags for the next line
            for (let j = 0; j < tagStack.length; j++) {
                currentLine += tagStack[j];
            }
        }
      }
    }
  }
  
  lines.push(currentLine);
  
  return lines;
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

  // Mask Triple Strings
  safeCode = safeCode.replace(/("""[\s\S]*?"""|'''[\s\S]*?''')/g, match => store(match, 'string'));
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
