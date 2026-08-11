# Isomorphism Definition

Formal statements and operational metrics for **MDI (Mathematical-Doctrinal
Isomorphism)**.

## Setup

Let:

- $\mathcal{T}$ be a space of legal texts (clauses, rules, statutory provisions, …).
- $\mathcal{S}$ be a mathematical structure space (labels, scores, embeddings, …).
- $f: \mathcal{T} \to \mathcal{S}$ a mapping.
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
