# MDI — Mathematical-Doctrinal Isomorphism

> **数理法理同构** — a unified mathematical relation between the
> doctrinal-normative space and the legal-application space.

**Version:** 0.1.6

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
