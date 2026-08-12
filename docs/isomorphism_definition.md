# Isomorphism Definition

Formal statements and operational metrics for **MDI (Mathematical-Doctrinal
Isomorphism)**.

## Core claim — a unified mathematical relation between two spaces

MDI asserts the existence of a mathematical construction $\Phi$ that relates the
**doctrinal-normative space** and the **legal-application space**:

- $\mathcal{N}$ — **doctrinal-normative space**: abstract, general legal rules /
  statutes / doctrines (norm, rule, provision).
- $\mathcal{A}$ — **legal-application space**: concrete, instantiated clauses /
  cases / application texts (clause, case, application).

$\Phi$ embeds both spaces into one shared vector space $\mathbb{R}^d$,
preserving doctrinal relations:

$$
\Phi = \left(\,g_N: \mathcal{N} \to \mathbb{R}^d,\ \ g_A: \mathcal{A} \to
\mathbb{R}^d\,\right)
$$

such that the doctrinal relation $R(n, a)$ between a norm $n \in \mathcal{N}$
and its application $a \in \mathcal{A}$ is stable with the embedding geometry:

$$
R(n,a)=\text{support / apply} \;\Rightarrow\;
d\big(g_N(n),\, g_A(a)\big)\ \text{is small}.
$$

**"Unified"** means: *one* shared embedding, *one* distance structure, carrying
both the normative and the application side — not one representation per task
type. The properties below (P1–P5) are the falsifiable operational forms of this
relation; each has a concrete metric.

## Setup

Working formulation for the properties: let

- $\mathcal{T}$ be a space of legal texts (norms and applications).
- $\mathcal{S}$ be a mathematical structure space (embeddings, labels, scores, …).
- $f: \mathcal{T} \to \mathcal{S}$ a mapping (an instance of $g_N$ / $g_A$).
- $d_\mathcal{S}$ a distance on $\mathcal{S}$.

MDI is the claim that there exist "doctrinal" maps $f$ satisfying the properties
below; the properties are falsifiable and each has an operational metric.

## Properties

### P1 · Well-definedness (paraphrase invariance)

**Statement.** If $t, t' \in \mathcal{T}$ are semantically equivalent
(same doctrinal content, different surface form), then
$f(t) = f(t')$.

**Metric.** Under a controlled paraphrase table (neutral synonym rewrites and
semantic rewrites), the fraction of pairs whose structural value changes.
Target: 0%.

### P2 · Isometry (doctrinal proximity)

**Statement.** Doctrinal equivalence is reflected in geometric proximity:
for premise–hypothesis pairs $(p, h)$ with a gold doctrinal relation
$y \in \{\text{entailment}, \text{contradiction}\}$,

$$
d_\mathcal{S}(f(p), f(h) \mid \text{entailment})
\;\ll\;
d_\mathcal{S}(f(p), f(h) \mid \text{contradiction}).
$$

**Metric.** Mann–Whitney U of the two distance distributions; report p.
(Note: for negation-type datasets, contradiction pairs may still be closer than
unrelated/neutral pairs — the core invariant is *entailment is the closest*.)

### P3 · Monotonicity

**Statement.** Adding a doctrinal severity/structure signal never decreases the
structural value: if $t'$ is $t$ plus a positive doctrinal signal, then
$f(t') \ge f(t)$.

**Metric.** Fraction of signal-boost pairs where the value decreased.
Target: 0.

### P4 · Traceability

**Statement.** Every component of $f(t)$ is evidenced by an original span of $t$.

**Metric.** Fraction of components whose evidence span appears verbatim in the
normalized source. Target: 100%.

### P5 · Lipschitz continuity (robustness)

**Statement.** For small semantic perturbations $\delta$ (synonymy, negation
injection), the drift is bounded:

$$
\|f(t + \delta) - f(t)\| \le L \cdot \|\delta\|.
$$

Content deletion is *not* a small perturbation — it causes proportionate,
information-driven change (sensitivity, not fragility).

**Metric.** p95 drift under micro-perturbations. Target: ~0; deletion p95
reported separately as content-sensitivity.

---

## Notes

- The properties are **task-agnostic**: they apply to any mapping from legal
  texts to a structure, independent of the downstream task.
- Observability of P2 depends on the representation granularity of $\mathcal{S}$;
  see `validation_report.md`.

---

## Positioning: theory as ruler, instantiation as competitor

(Established 2026-08-12; prevents a recurring category-error in reading the
evaluations.)

MDI has **two distinct components** that must not be conflated:

1. **MDI the theory** — the existence claim (dual-space $\Phi$, P1–P5). This is
   a *ruler*: it defines measurable properties of any representation space
   (isometry fidelity, structure, spectral compression). A theory is not a
   competitor; it is the measuring instrument. Like Vista's effective-rank /
   spectral-entropy analysis tools, the theory's value is *diagnostic*, not
   "winning" against models.
2. **MDI-φ the instantiation** — a concrete representation
   $\phi(x) = xW$ (base encoder + learned projection). This *is* a competitor:
   a text→vector representation function, same layer as tfidf / minilm / mpnet
   / legalbert.

**Consequence for evaluation.**

- Representation-level comparisons (MDI-φ vs common representations on
  isometry/structure AUC) are **valid and standard** — all are representation
  methods at the same layer.
- Downstream evaluation (representation as features + a *fixed* simple
  classifier, e.g. LR) is the standard representation-as-features paradigm.
- The theory's P1–P5 properties are never "compared"; they are the **tool** that
  measures whether a space (any space) carries the doctrinal structure.

This two-part reading is fixed documentation: the representation competes,
the theory measures.

---

## Standard input / output of MDI

**Theory level (the formal object).**

| | Definition |
|---|---|
| **Input** | Dual spaces + their relation: the doctrinal-normative space $\mathcal{N}$ (rules / statutes / doctrines) and the legal-application space $\mathcal{A}$ (clauses / cases / statements), together with a **doctrinal alignment relation** $\mathcal{R} \subseteq \mathcal{N}\times\mathcal{A}$ (e.g. $n$ supports/applies to $a$ ⇒ entailment; $n$ contradicts $a$). |
| **Output** | An embedding $\Phi=(g_{\mathcal{N}}, g_{\mathcal{A}})$ into a **shared** vector space such that $\mathcal{R}$ is geometrically preserved: $(n,a)\in\mathcal{R}^+\Rightarrow d(g_{\mathcal{N}}(n),g_{\mathcal{A}}(a))$ small, $(n,a)\in\mathcal{R}^-\Rightarrow$ large, satisfying P1–P5. |

**Instantiation level (MDI-φ, operational).**

| | Definition |
|---|---|
| **Input** | Text pairs $(p,h)$ + relation label $y\in\{E,N,C\}$ (entailment / neutral / contradiction); $p$ drawn from $\mathcal{N}$, $h$ from $\mathcal{A}$. |
| **Output** | A linear projection $W$ (768→64) mapping every text to a 64-d *doctrinal vector*, so that the distance order $E<N<C$ holds. |

**Key point.** MDI's output is *not* a class or a score — it is **a space together with a preserved relation**. Downstream usage (classifier / retrieval / similarity) lives on top of the representation and is not part of MDI's output.

## Definition of "alignment-type" (alignment-oriented) tasks

An **operational, non-circular** definition based on the structure of the
ground truth:

> **Task $T$ is alignment-type ⟺ its ground truth can be formalized as a
> *relation between two texts* (entailment / support / applies / contradiction /
> equivalence), NOT as a *class membership of a single text*.**

**Decision procedure (three questions):**
1. **Input** — is the input a *pair of texts* or a *single text*? Pair ⇒ candidate; single ⇒ non-alignment.
2. **Label** — does the label express a *relation* (E/N/C ordering) or a *category* (class id)? Relation ⇒ alignment; category ⇒ non-alignment.
3. **Preservability** — can the relation be geometrically preserved (isometry: entailment close, contradiction far)? Embeddable ⇒ alignment; pure cluster structure ⇒ non-alignment.

**Classification of the benchmark tasks:**

| Task | Structure | Type |
|---|---|---|
| NLI / isometry (ContractNLI, WillsNLI) | text pair + E/N/C order | ✅ alignment-type (MDI's direct target) |
| Retrieval (premise → hypothesis top-k) | text pair + relative rank | ⚠️ partial (implicit similarity, not strict entailment order) |
| Classification (CUAD/MAUD/SCOTUS/LEDGAR) | single text + class label | ❌ non-alignment |
| Generation / summarization / QA extraction | sequence output | ❌ non-alignment |

**Why this matters.** It turns MDI's applicability boundary from "feeling" into
a *decision* — answer the three questions and you know whether MDI-φ can bring
gain. It also explains the *alignment bias*: φ's supervision (the E<N<C order)
is exactly the ground-truth form of alignment-type tasks, so its gain is
concentrated there; non-alignment tasks have a different ground-truth form, so
MDI-φ carries no advantage (it is still usable, but not better).

---

## Geometricity of MDI-φ and the algebraization route

The "similarity" use of MDI-φ (distance / feature augmentation) is the most
elementary way to consume the *new-kind information* MDI creates (the doctrinal
alignment structure — qualitatively new vs. the semantic-similarity information
of off-the-shelf embeddings). Algebraizing MDI — using the space *operationally*
(vector-space algebra, cross-space mapping, structure preservation) instead of
reading distances — is treated as **the problem to be solved next**, not a
deferred future-work paragraph.

**Prerequisite: establishing geometricity** (why algebraic-geometry methods may
apply). Empirically checked on the training domain (n=1472; `verify_geometry.py`
):

**A. Spectral compression (Vista-style):**

| Space | effective rank | spectral entropy | PCA-95% | isometry AUC |
|---|---|---|---|---|
| mpnet (768) | 313.0 | 5.75 | 157 | 0.391 |
| **MDI-φ (64)** | **49.3** | **3.90** | **45** | **0.285** |

→ φ concentrates the doctrinal structure into a **low-dimensional subspace**
(45 dims carry what mpnet needs 157 for; effective rank 49 vs 313). Low-rank,
low-dimensional structure is exactly the condition under which a space is
approximable by algebraic varieties / polynomials.

**B. Structural form (E vs C separability, 3-fold SVM):**

| Representation | linear | poly-2 | poly-3 | RBF |
|---|---|---|---|---|
| mpnet (diff) | 0.609 | 0.636 | 0.612 | 0.647 |
| **φ (diff)** | **0.638** | 0.649 | 0.617 | 0.635 |
| **φ (concat)** | 0.634 | **0.650** | 0.625 | **0.663** |

→ the doctrinal relation in φ is **cleaner and more linearly separable than in
mpnet** (0.638 vs 0.609 on the difference representation), with only marginal
polynomial gain (poly-2 +0.011; poly-3 *drops*).

**Geometricity conclusion.** MDI-φ is geometric: low-dimensional, isometry-
preserving, and the E/C relation is dominantly **affine / linear** in structure
rather than a complex non-linear algebraic variety. **Algebraic-geometry methods
are admissible** (the space is low-rank and variety-approximable), but the
empirical form points to **linear / affine algebra and lattice-theoretic
structure first** (subspaces, linear maps, order structures), before higher-
degree polynomial varieties — because the data shows the relation is mostly
linear, and only a small non-linear residue remains.

### Algebraization test (does the relation have algebraic structure?)

Empirically tested (`verify_algebra.py`, training domain n=1472, phi-space):

**A. Polynomial map norm→application (Ridge RMSE, 70/30):**

| | deg-1 | deg-2 | deg-3 |
|---|---|---|---|
| entailment | **0.0317** | 0.0351 | 0.0311 |
| contradiction | **0.0335** | 0.0341 | — (few samples) |

→ **no polynomial gain**: deg-2 worse, deg-3 ≈ linear. The norm→application
map is **linear**; non-linear polynomial terms carry no extra information.

**B. Relation-vector subspace (PCA of d = h−p):**

| difference vector | top-5 | top-10 | top-20 |
|---|---|---|---|
| entailment-d | 0.389 | 0.562 | 0.757 |
| contradiction-d | 0.446 | 0.623 | 0.799 |
| **direction alignment** | \|cos(v_E, v_C)\| = **0.876** | | |

→ the entailment and contradiction difference vectors point in **almost the
same direction** — the doctrinal relation is **translation-like** (same
direction, different magnitude), a strong signature of **linear / affine**
structure rather than distinct geometric regions.

**C. Polynomial separability (E vs C, logistic 5-fold):**

| poly-1 | poly-2 | poly-3 |
|---|---|---|
| **0.643** | 0.606 | 0.612 |

→ **linear is best**; polynomial features hurt. The classes are separated by a
**linear hyperplane**, not an algebraic hypersurface.

**Algebraization conclusion (empirical correction of the algebraic-geometry
hypothesis).** The doctrinal relation in MDI-φ is **linear / affine algebraic
structure, not a polynomial algebraic variety**:

- polynomial map: no gain → relation is a linear operator (A)
- difference vectors aligned (|cos|=0.876) → translation/affine (B)
- linear separability optimal → linear hyperplane (C)

The **spirit of algebraization holds** (the space is a structured algebraic
object, not raw similarity), but the correct toolkit is **linear algebra +
affine geometry + lattice/order theory**, not full algebraic geometry
(polynomial ideals, varieties, hypersurfaces). Concretely:

1. **Affine-subspace model** — the relation lives on an affine subspace
   (translation + linear map), directly analyzable by linear algebra.
2. **Linear operator on norm→application** — confirmed; enables operator
   algebra / eigendecomposition of the doctrinal transform.
3. **Lattice/order structure** — the E<N<C ordering is a partial order, the
   natural domain of lattice theory, going beyond mere geometry.
