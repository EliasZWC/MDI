# MDI — Mathematical-Doctrinal Isomorphism

> **数理法理同构** — a general theory of structure-preserving mappings from
> legal texts to mathematical structures.

**Version:** 0.1.0

**MDI** posits that there exists a mapping
$f: \text{legal text} \to \text{mathematical structure}$ that preserves
doctrinal structure. The theory is **independent of any downstream task** —
classification, entailment, judgment prediction, and other legal tasks are all
*instances* of this isomorphism; none is privileged.

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

The theory is validated on the **natural entailment annotations** of public legal
NLI datasets, using only generic representations:

| Domain (dataset) | Isomorphism type | Representation | E<C significance |
|---|---|---|---|
| Contract (ContractNLI) | lexical | TF-IDF | p = 2.45e-05 ✓ |
| Wills (WillsNLI) | negation | semantic embedding (all-MiniLM) | p = 0.0185 ✓ |
| Tax (SARA) | computational | — | n.s. |

**Invariant tested:** entailment pairs are significantly closer in the
representation space than contradiction pairs.

**Finding:** representation granularity determines the observability of the
isomorphism — lexical spaces carry lexical isomorphism, universal semantic
embeddings carry negation-type isomorphism, computational isomorphism requires a
reasoning-capable representation. This motivates *doctrinal-sensitive*
representations as the general carrier of legal isomorphism.

Run the validation:

```bash
pip install -r requirements.txt
python code/verify_cross_domain.py --max 600
```

---

## Repository layout

```
MDI/
  README.md                          this overview
  docs/isomorphism_definition.md     formal definitions & operational metrics
  docs/validation_report.md          full cross-domain evidence
  code/verify_cross_domain.py        self-contained cross-domain validator
  requirements.txt
```

## License

MIT
