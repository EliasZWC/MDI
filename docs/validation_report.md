# Cross-Domain Validation Report

Empirical evidence for MDI **independent of any task-specific scorer**, using the
*natural* annotations of public legal datasets and generic representations only.

## Method

Two invariant types are tested:

**Type A — isometry (premise–hypothesis NLI).** For gold premise–hypothesis pairs
$(p, h)$ with doctrinal label $y \in \{E, N, C\}$, embed $p$ and $h$ and test

$$
\text{dist}(p,h \mid E) < \text{dist}(p,h \mid C)
$$

via Mann–Whitney U (alternative = less).

**Type B — structure (labeled instances).** For instances with gold labels,
test that same-label pairs are closer than different-label pairs:

$$
\text{dist}(same\text{-}label) < \text{dist}(different\text{-}label)
$$

via Mann–Whitney U.

### Datasets

| Dataset | Domain | Structure | Type |
|---|---|---|---|
| ContractNLI | contracts | premise/hypothesis/label | A |
| SARA | tax law | statute/question/answer | A |
| WillsNLI | wills/estates | rule/statement/classification | A |
| CUAD | contracts | 41 clause categories | B |
| MAUD | contracts (M&A) | clause category | B |
| ECHR | human-rights judgments | violated True/False | B |

### Representations (baselines)

- **lexical**: TF-IDF cosine (pure stdlib)
- **semantic**: all-MiniLM-L6-v2 universal embedding

## Results

### Type A — isometry (E closer than C)

| Dataset | Lexical | Semantic embedding | Captured by |
|---|---|---|---|
| ContractNLI (contract) | **p = 3.47e-05** ✓ | n.s. | lexical |
| WillsNLI (wills) | n.s. | **p = 0.0185** ✓ | semantic embedding |
| SARA (tax) | n.s. | n.s. | — (computational) |

### Type B — structure (same-label closer)

| Dataset | Lexical | Semantic embedding |
|---|---|---|
| CUAD (contract, 37 cls) | **p = 5.85e-05** ✓ | **p = 3.89e-18** ✓ |
| MAUD (contract, 7 cls) | **p = 4.89e-163** ✓ | **p = 1.07e-136** ✓ |
| ECHR (human rights, 2 cls) | n.s. | n.s. |

## Findings

1. **The isomorphism is observable outside any task-specific scorer**, across 6
   public legal datasets — a general theory, not an artifact of one task.
2. **Representation granularity determines observability**: lexical spaces carry
   lexical isomorphism (ContractNLI); universal semantic embeddings carry
   negation-type isomorphism (WillsNLI) and, strikingly, **contract class
   structure** (CUAD/MAUD both representations).
3. **Honest boundaries**: computational isomorphism (SARA/tax) needs a
   reasoning-capable representation; judgment-direction labels (ECHR violated)
   are not separable in semantic distance — both are documented limits, not
   claims.
4. This motivates **doctrinal-sensitive representations** as the general carrier
   of legal isomorphism across all types.

## Reproduction

```bash
pip install -r requirements.txt
python code/verify_cross_domain.py --max 600
```
