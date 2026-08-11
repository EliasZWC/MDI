# Cross-Domain Validation Report

Empirical evidence for MDI **independent of any task-specific scorer**, using the
*natural* entailment annotations of public legal NLI datasets and generic
representations only.

## Method

For each dataset we take its gold premise–hypothesis pairs $(p, h)$ with
doctrinal label $y \in \{\text{entailment (E)}, \text{neutral (N)},
\text{contradiction (C)}\}$, embed $p$ and $h$ with a generic representation,
and test the invariant

$$
\text{dist}(p,h \mid E) < \text{dist}(p,h \mid C)
$$

via Mann–Whitney U.

### Datasets

| Dataset | Domain | Structure | n (used) |
|---|---|---|---|
| ContractNLI | contracts | premise/hypothesis/label | 600 |
| SARA | tax law | statute/question/answer | 272 |
| WillsNLI | wills/estates | rule/statement/classification (1=support, 0=refute, 2=unrelated) | 600 |

### Representations

- **lexical**: TF-IDF cosine (pure stdlib)
- **semantic**: all-MiniLM-L6-v2 universal embedding

## Results

| Domain (dataset) | Isomorphism type | Lexical | Semantic embedding | Captured by |
|---|---|---|---|---|
| Contract (ContractNLI) | lexical | **p = 2.45e-05** ✓ | n.s. | lexical |
| Wills (WillsNLI) | negation | n.s. | **p = 0.0185** ✓ | semantic embedding |
| Tax (SARA) | computational | n.s. | n.s. | — |

## Findings

1. **The isomorphism is observable outside any task-specific scorer** — a general
   theory, not an artifact of one task.
2. **Representation granularity determines observability**: lexical spaces carry
   lexical isomorphism (contracts); universal semantic embeddings carry
   negation-type isomorphism (wills); computational isomorphism (tax) needs a
   reasoning-capable representation that neither generic space provides.
3. This motivates **doctrinal-sensitive representations** as the general carrier
   of legal isomorphism across all types.

## Reproduction

```bash
pip install -r requirements.txt
python code/verify_cross_domain.py --max 600
```
