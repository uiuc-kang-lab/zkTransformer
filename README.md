# zkTransformer

Artifact for the paper *zkTransformer: Scalable Zero-Knowledge Proofs for LLM
Inference via Einsum Arithmetization and Structured Exponentiation Lookups*
([zkTransformer.pdf](zkTransformer.pdf)).

zkTransformer proves transformer inference end-to-end, from input token IDs
to output token IDs. It combines **EinSumcheck**, which compiles Einsum-expressible
linear ops into a single Sumcheck, with **StructuredExp**, which handles
exponentiation lookups without committing to the table. The polynomial
commitment scheme is KZH-3 over BN254.

## Layout

```
src/
├── basicblock/     # Building blocks (Einsum, Add, Exp, Range, Permute, ...)
├── crypto/         # KZH-3 polynomial commitments, Sumcheck prover/verifier
├── dag/            # Circuit builder and model definitions (GPT-2, LLaMA, nanoGPT)
├── util/
└── bin/            # Proving binaries used in the paper (see below)
ppl_repro/          # Perplexity scripts for Table 1
parse_timing.py     # Sums commit times from a proving log
parse_prove_timing.py  # Per-operation prove times from a proving log (Table 7)
```

## Build

The paper runs used Rust `nightly-2025-07-01` with the default features
(BN254, arkworks backend) on a 32-core Intel Xeon Platinum 8358 server with
2 TB of memory.

```bash
cargo +nightly-2025-07-01 build --release
```

Each binary checks for `<size>.srs` files in the working directory. Any that
are missing are generated and cached there on first use. To pre-generate one
SRS by hand:

```bash
cargo +nightly-2025-07-01 run --release --bin setup -- generate <log_size>
```

Each run prints the commit, prove and verify times, the proof size, and
whether verification passed. Use `RUST_LOG=debug` to get the per-node timing
lines that the log parsers read.

## Reproducing the paper

All models use random weights with the real architecture shapes. Prover cost
depends only on the shapes. Each binary is configured through environment
variables.

| Paper result | Binary | Settings |
|---|---|---|
| Table 2, Table 6 (full system), Table 7, Table 8: GPT-2, 16 prompt + 16 generated tokens | `oneshot_gpt2` | `SEQ_LEN=32 PROMPT_LEN=16 VOCAB_SIZE=50257` |
| Figure 1: GPT-2 seq-length scaling | `oneshot_gpt2` | `SEQ_LEN∈{32,64,128,256,512,1024} PROMPT_LEN=16 VOCAB_SIZE=50257` |
| Table 3: GPT-2 single token, 64 input tokens | `oneshot_gpt2` / `gpt2` | `SEQ_LEN=64` (`PROMPT_LEN=64` for `oneshot_gpt2`) |
| Table 4: nanoGPT (EZKL config), 1 token | `nanogpt` | `SEQ_LEN=1` |
| Table 5: LLaMA2-7B single token | `oneshot_llama` / `llama` | `NUM_LAYERS=32` (`VOCAB_SIZE=32000` for `oneshot_llama`) |
| Table 1: perplexity before/after quantization | `ppl_repro/` | see [ppl_repro/README.md](ppl_repro/README.md) |

Example:

```bash
SEQ_LEN=32 PROMPT_LEN=16 VOCAB_SIZE=50257 RUST_LOG=debug \
  cargo +nightly-2025-07-01 run --release --bin oneshot_gpt2 2>&1 | tee gpt2_32.log
python parse_prove_timing.py gpt2_32.log   # per-operation prove-time breakdown
python parse_timing.py gpt2_32.log         # commit-time totals
```

### Binaries

- `oneshot_gpt2`: GPT-2 Small, full pipeline: token embedding, positional
  encoding, 12 transformer blocks, LM head and argmax check. Env vars:
  `SEQ_LEN`, `PROMPT_LEN` (default `SEQ_LEN/2`), `VOCAB_SIZE`, and
  `SKIP_AUTOREGRESSIVE=1`, which uses random tokens and skips the AR
  generation loop. Proving time does not change with it.
- `oneshot_llama`: LLaMA2-7B, full pipeline with RoPE and a separate LM
  head. Env vars: `SEQ_LEN`, `PROMPT_LEN`, `VOCAB_SIZE`, `NUM_LAYERS`,
  `NUM_HEADS`, `HEAD_DIM`, `MLP_DIM`, `SKIP_AUTOREGRESSIVE`.
- `gpt2`, `llama`: transformer blocks only, on a pre-embedded input.
  Env vars: `SEQ_LEN`, `SEED`, and `NUM_LAYERS` for `llama`.
- `nanogpt`: the nanoGPT model EZKL uses by default (4 layers, 4 heads,
  n_embd=64). Env var: `SEQ_LEN`.
- `structured_exp`, `nonstructured_exp`: exponentiation with and without
  StructuredExp.

### Baselines

- ZKML: https://github.com/uiuc-kang-lab/zkml.git
- zkLLM: https://github.com/jvhs0706/zkllm-ccs2024.git
- zkGPT: https://zenodo.org/records/16958213

## License

Apache-2.0
