# MDI — Mathematical-Doctrinal Isomorphism

> **数理法理同构** — a unified mathematical relation between the
> doctrinal-normative space and the legal-application space.

**Version:** 0.3.0

**MDI** asserts the existence of a mathematical construction $\Phi$ that embeds
both the **doctrinal-normative space** $\mathcal{N}$ (abstract rules, statutes,
doctrines) and the **legal-application space** $\mathcal{A}$ (concrete clauses,
cases, applications) into a **shared** vector space, preserving doctrinal
relations: a norm $n$ that supports/applies to an application $a$ maps to
nearby points, $d(g_N(n), g_A(a))$ small. *One* embedding, *one* distance
structure — not one representation per task type.

**MDI is a paradigm** — 数理法理同构 — with **two standard steps**:
(1) **Mapping (映射)**: build the Doctrinal Space — φ projects texts/norms into
a unified computable representation space; (2) **Channel (通道)**: operate on
that space — fit different mathematical treatments (alignment / category /
metric / kernel / subspace) and select the one that fits a given task. Both
steps belong to the paradigm: mapping builds the space, channel uses it.
Validating MDI = validating the whole flow (mapping quality + channel
effectiveness).

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
3. **L3 — augment the strong:** `legalbert+phi` augments legal-bert on
   classification: SCOTUS 0.582→**0.593**, LEDGAR 0.547→**0.571**, CUAD
   0.633→**0.640** (MAUD 0.990→0.985, near-parity, honest boundary). Verified
   with the current v0.3.0 weights (see `code/mdi_version.py`); log header
   records `version=0.3.0` + weight filenames.

MDI-φ is also the **isometry leader** on both Type-A domains (ContractNLI
0.344, WillsNLI 0.402 — best of all 6 representations) and the best structure
representation on ECHR (0.464). See
[`docs/validation_report.md`](docs/validation_report.md) §v0.2.0 for full
8-dataset × 6-representation tables.

### v0.3.0 — MDI as paradigm: Mapping + Channel (拟合)

**This is the current framework version.** MDI is formalized as a **two-step
paradigm**, and both steps are *fitted/selected per dataset* — this is what
makes MDI a paradigm, not a single projection:

- **Step 1 — Mapping (映射)**: φ projects texts/norms into a unified
  computable Doctrinal Space. The projection is trained with the theory's own
  axioms in the objective (P2 isometry + P3 E<N<C monotonicity + P5 Lipschitz
  spectral-norm bound), producing `mdi_W_v2b_mpnet.npy` (768→64, ‖W‖₂=2.0).
- **Step 2 — Channel (通道)**: on that space we *fit and select* a
  mathematical treatment per task — alignment / category (classification) /
  metric — choosing the channel that fits each dataset. Validating MDI =
  validating the whole flow (mapping quality × channel effectiveness).

This version supersedes v0.2.5's "MDI = mapping only" framing (mapping alone
was a representation; adding the fitted channel makes it a paradigm).

```bash
python code/mdi_phi_v2.py --model all-mpnet-base-v2 --out mdi_W_v2b_mpnet.npy --lip 2.0
python code/post_mapping.py --W mdi_W_v2b_mpnet.npy        # fit-and-select channels
python code/channel_close.py --W mdi_W_v2b_mpnet.npy       # closed-loop per-dataset channel
```

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

> **v0.2.4 correction.** Re-evaluating MDI-φ v2b with `--only` (single-dataset
> protocol, `eval_unified.py --only`) exposed that the earlier full-run numbers
> (0.300 / 0.386) used a stale weight (v2 vs v2b). The accurate v2b numbers are
> **stronger**: ContractNLI **0.215** (d=1.18, leader by −0.139 over tfidf),
> WillsNLI **0.338** (d=0.63, leader by −0.065 over minilm), ECHR **0.462**
> (still leader). The single-dataset protocol is now the standard for baseline
> comparisons (faster, and removes version-mixups).

### v0.2.2 — geometricity & algebraization

**Geometricity** (`verify_geometry.py`): MDI-φ concentrates the doctrinal
structure into a **low-dimensional subspace** — effective rank 49 vs 313,
PCA-95% dims 45 vs 157 (mpnet), spectral entropy 3.90 vs 5.75 — and is
**linear-separable** on the doctrinal relation (E/C SVM 0.638 vs mpnet 0.609).

**Algebraization** (`verify_algebra.py`): the doctrinal relation is **linear /
affine algebraic structure, not a polynomial variety** —
- norm→application map: linear RMSE 0.0317 ≈ poly-3 0.0311 (poly-2 worse) → linear operator
- entailment/contradiction difference vectors aligned (|cos(v_E,v_C)|=0.876) → translation/affine
- E/C linear separability 0.643 > poly-2 0.606 > poly-3 0.612 → linear hyperplane

So algebraization proceeds with **linear algebra + affine geometry + lattice /
order theory** (the E<N<C order is a partial order), not full algebraic
geometry. Input/output definition and the alignment-type decision procedure
are formalized in [`docs/isomorphism_definition.md`](docs/isomorphism_definition.md).

### v0.2.3 — algebraic applications (using the new-kind info operationally)

The algebraization facts are turned into **operations**, not distance reading
(`apply_algebra.py`):

| Application | Operation | Result |
|---|---|---|
| **APPL-1 cross-space mapping → retrieval/recommendation** | train linear map norm→application; for a held-out norm predict the application vector and retrieve nearest real applications | top-5 hit **0.044** (13× random), top-10 **0.081** (12×) |
| **APPL-2 difference-vector algebra → relation classifier** | linear classifier on d = h−p predicts the doctrinal order E/N/C | 3-way acc **0.615** vs majority 0.408 |
| **APPL-3 entailment translation → completion** | d_E is a stable "entailment translation": subtracting it from an entailment hypothesis brings it near its premise | entail drift 0.391 < non-entail 0.443 |

All three hold: the doctrinal relation is **operable** (mapping generates,
vector algebra discriminates, translation completes) — the new-kind
information MDI creates is consumed *computationally*, not just measured.

```bash
python code/apply_algebra.py --data-dir <dir> --W mdi_W_v2b_mpnet.npy
```

### v0.2.4 — layered architecture (structure in φ, operations on top)

Attempts to load the *translation consistency* (APPL-3's algebraic fact) into
the training objective failed in both forms — `mdi_phi_v3.py` (projection
share) and `mdi_phi_v31.py` (cosine alignment) — breaking P3 (dN<dE) and
degrading isometry (AUC 0.285→0.395/0.424), even though APPL-1 retrieval
improved (top-1 0.006→0.019). This is a genuine trade-off, not a bug: direction
alignment and amplitude ordering (P3) conflict in one objective.

**Conclusion — MDI is consumed in layers:** φ keeps the structure (v2b:
AUC 0.285, full E<N<C), and the operations live in the application layer
(APPL-1/2/3). "Enhancing MDI via advanced applications" works by strengthening
the application layer, not by altering φ. Also corrected the v2b baseline
numbers via the single-dataset protocol (`--only`): ContractNLI 0.215 (d=1.18),
WillsNLI 0.338 (d=0.63), ECHR 0.462.

```bash
python code/eval_unified.py --W mdi_W_v2b_mpnet.npy --mdi-base all-mpnet-base-v2 --only ContractNLI
```

```bash
python code/verify_geometry.py --data-dir <dir> --W mdi_W_v2b_mpnet.npy
python code/verify_algebra.py   --data-dir <dir> --W mdi_W_v2b_mpnet.npy
```

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
