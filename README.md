# MDI — Mathematical-Doctrinal Isomorphism

> **数理法理同构** — a unified mathematical relation between the
> doctrinal-normative space and the legal-application space.

**Version:** 0.2.1

**MDI** asserts the existence of a mathematical construction $\Phi$ that embeds
both the **doctrinal-normative space** $\mathcal{N}$ (abstract rules, statutes,
doctrines) and the **legal-application space** $\mathcal{A}$ (concrete clauses,
cases, applications) into a **shared** vector space, preserving doctrinal
relations: a norm $n$ that supports/applies to an application $a$ maps to
nearby points, $d(g_N(n), g_A(a))$ small. *One* embedding, *one* distance
structure — not one representation per task type.

---

## Isomorphism properties

| Property | Statement |
|---|---|
| **Well-definedness** | semantically equivalent texts map to the same structural point (paraphrase-invariance) |
| **Isometry** | doctrinal equivalence ⟺ geometric proximity: entailment pairs are significantly closer than contradiction pairs |
| **Monotonicity** | adding a doctrinal signal never decreases the structural value |
| **Traceability** | every structural component is evidenced by an original span |
| **Lipschitz continuity** | micro semantic perturbations cause bounded structural drift; content deletion causes proportionate, information-driven change |

See [`docs/isomorphism_definition.md`](docs/isomorphism_definition.md) for the
formal statements and their operational metrics.

---

## Cross-domain validation (no task-specific scorer involved)

The theory is validated on the **natural annotations** of public legal datasets
(6 datasets, 2 invariant types), using only generic representations:

**Type A — isometry** (entailment closer than contradiction):

| Domain (dataset) | tfidf | bigram | minilm | mpnet | legalbert¹ |
|---|---|---|---|---|---|
| Contract (ContractNLI) | **3.47e-05** ✓ | **3.09e-04** ✓ | n.s. | **7.79e-04** ✓ | n.s. |
| Wills (WillsNLI) | n.s. | n.s. | **0.0185** ✓ | n.s. | n.s. |
| Tax (SARA) | n.s. | n.s. | n.s. | n.s. | n.s. (boundary) |

**Type B — structure** (same-label closer than different-label):

| Domain (dataset) | tfidf | bigram | minilm | mpnet | legalbert¹ |
|---|---|---|---|---|---|
| Contract (CUAD, 37 cls) | **5.85e-05** ✓ | **2.88e-05** ✓ | **3.89e-18** ✓ | **3.54e-19** ✓ | **4.56e-13** ✓ |
| Contract (MAUD) | **4.89e-163** ✓ | **1.37e-162** ✓ | **1.07e-136** ✓ | **7.80e-133** ✓ | **1.76e-73** ✓ |
| Human-rights (ECHR) | n.s. | n.s. | n.s. | **1.43e-02** ✓ | **1.76e-02** ✓ |

¹ legal-bert run with `--max 300`; others `--max 600`.

**Finding:** there is no free representation — the 5×6 matrix shows each
representation covers a different subset of isomorphism types (lexical covers
contract isometry, minilm covers wills isometry, mpnet adds ContractNLI + ECHR,
legal-bert covers all Type-B structure but no Type-A isometry). **Legal-specific
representation does not dominate generic ones**; contract class structure holds
under all five; computational tax isomorphism (SARA) is a documented boundary.
This motivates *doctrinal-sensitive* representations as the general carrier of
legal isomorphism.

Run the validation:

```bash
pip install -r requirements.txt
python code/verify_cross_domain.py --data-dir <dir> --max 600 --models tfidf,bigram,minilm,mpnet,legalbert
```

---

## The unified representation (MDI-φ) and three-layer utility

MDI's own construction is realized as a **learned linear projection**
$\phi(x)=xW$ into a low-dimensional *doctrinal space* (64-d, trained by
contrastive alignment on normative–application pairs from ContractNLI /
WillsNLI / SARA). This gives a **three-layer contribution**:

1. **L1 — interpretability (all models):** $\phi$ is a low-rank linear map;
   the 64-d doctrinal vector and its alignment geometry are inspectable and
   evidence-linked — interpretability is inherent, not bolted on.
2. **L2 — lift the simple:** in the alignment domain (ContractNLI NLI) the 64-d
   φ reaches **0.660** vs 768-d mpnet 0.630 — a 1/12-dimensional projection
   *exceeding* its base model on the doctrinal relation task.
3. **L3 — augment the strong:** `legalbert+phi` improves legal-bert on all
   three classification tasks (SCOTUS 0.582→0.590, LEDGAR 0.580→0.598,
   CUAD 0.676→0.680); `mpnet+phi` is neutral (no loss).

MDI-φ is also the **isometry leader** on both Type-A domains (ContractNLI
0.344, WillsNLI 0.402 — best of all 6 representations) and the best structure
representation on ECHR (0.464). See
[`docs/validation_report.md`](docs/validation_report.md) §v0.2.0 for full
8-dataset × 6-representation tables.

### v0.2.1 — theory-loaded MDI-φ (P2+P3+P5 in the objective)

The depth application: instead of a bare contrastive projection, the theory's
own axioms now enter the training objective. P3 (monotonicity) adds a
per-triplet E<N<C constraint; P5 (Lipschitz) is enforced by explicit spectral
normalization ($\lVert W\rVert_2 \le L=2.0$).

| Evidence (train domain) | v1 (bare hinge) | v2 (theory-loaded) |
|---|---|---|
| isometry AUC | 0.406 | **0.361** |
| \|\|W\|\|₂ (Lipschitz) | 3.520 | **2.000** |
| \|\|W\|\|F | 22.1 | **12.6** |
| dE / dN / dC | 0.67/0.83/0.71 | **0.41/0.47/0.44** |

Cross-domain, v2 is the isometry leader and *improves on v1*: ContractNLI
**0.300** (v1 0.344), WillsNLI **0.386** (v1 0.402), ECHR **0.463** — while
satisfying P5 (half the Lipschitz constant) and leaving Type-B structure
unchanged. The axioms help, they do not hurt. See
[`docs/validation_report.md`](docs/validation_report.md) §v0.2.1.

```bash
python code/mdi_unified.py --model all-mpnet-base-v2 --out mdi_W_mpnet.npy
python code/eval_unified.py --W mdi_W_mpnet.npy --mdi-base all-mpnet-base-v2
python code/eval_downstream.py --data-dir <dir> --cv 3
python code/mdi_phi_v2.py --model all-mpnet-base-v2 --out mdi_W_v2_mpnet.npy --lip 2.0
python code/eval_unified.py --W mdi_W_v2_mpnet.npy --mdi-base all-mpnet-base-v2
```

---

## Repository layout

```
MDI/
  README.md                          this overview
  docs/isomorphism_definition.md     formal definitions & operational metrics
  docs/validation_report.md          full cross-domain evidence
  code/verify_cross_domain.py        self-contained cross-domain validator
  code/mdi_unified.py                trains the MDI-φ projection (v1)
  code/mdi_phi_v2.py                 theory-loaded MDI-φ (P2+P3+P5)
  code/verify_phi_v2.py              verifies P3/P5/P2 on the projection
  code/eval_unified.py               8-dataset × 6-representation isometry/structure
  code/eval_downstream.py            three-layer utility (L1/L2/L3) downstream
  requirements.txt
```

## License

MIT
