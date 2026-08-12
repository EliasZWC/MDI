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
