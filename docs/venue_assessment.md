# Venue Assessment — MDI (Mathematical-Doctrinal Isomorphism)

**Date:** 2026-08-12 (rev. 2 — CCF-aware)
**Scope:** **full-paper assumption** — theorem formalization (P1–P5) and
ablation studies are treated as **completed** (user: both will be done).
Assessment dimension = **CCF recommended venues** (user requirement).
**Version:** 0.2.0 evidence base + planned theorem/ablation.

---

## 0. Correction from rev. 1

- Rev. 1 scored domain *fit* only and missed the **CCF dimension**; it also
  treated theorem/ablation as missing. Both were wrong.
- NLLP / ICAIL / JURIX are domain-specialist venues (real prestige in legal AI)
  but are **NOT CCF-ranked**. They are demoted to "domain-alternate" below.
- CCF ranks per CCF 2022 recommended list; verify officially before scheduling.

---

## 1. Evidence profile (what the paper sells)

| Asset | Strength | Honest caveat |
|---|---|---|
| Theoretical novelty | Strong — dual-space Φ + P1–P5 isomorphism, first-of-kind framing (jurimetrics = statistics, legal embeddings = no isomorphism claim, deontic logic = non-geometric) | Formal theorem statement deferred; claims rest on empirical regularity |
| Method (MDI-φ) | Linear, low-rank (768→64), interpretable, traceable (L1) | Not a representation-learning SOTA; a 64-d projection |
| Cross-domain consistency | 8 datasets; isometry leader on ContractNLI (0.344) & WillsNLI (0.402); ECHR structure 0.464 | Not optimal on CUAD/MAUD/LEDGAR structure (alignment signal ≠ label-feature signal) |
| Downstream utility | L2: 64-d φ 0.660 > mpnet 0.630 (ContractNLI NLI); L3: legalbert+phi up on 3/3 classification (small, consistent) | Gains small (+0.004~0.018); domain-dependent; mpnet+phi neutral |

**Net positioning (full-paper):** a *formalized theory + interpretable
representation + cross-domain consistency + ablation-supported* paper — with
P1–P5 as theorems and a complete ablation suite, it is no longer a
"empirical-regularity-only" paper. The selling point is the theory, not SOTA.

---

## 2. CCF status of relevant venues

| Venue | Type | CCF rank | Note |
|---|---|---|---|
| ACL | main conf | **A** | NLP flagship |
| AAAI | main conf | **A** | AI generalist |
| IJCAI | main conf | **A** | AI generalist, biennial |
| EMNLP | main conf | **B** | NLP, annual (~Nov) |
| NAACL | main conf | **B** | NLP, annual (~spring) |
| COLING | main conf | **B** | NLP, biennial |
| ECAI | main conf | **B** | AI, Europe, biennial |
| NLLP | workshop | not listed | EMNLP workshop (domain-specialist, high fit, no CCF rank) |
| ICAIL | main conf | not listed | ACM legal-AI (domain-specialist) |
| JURIX | main conf | not listed | European legal-KB |

**The three previously recommended (NLLP / ICAIL / JURIX) are domain-specialist
venues — real prestige in legal AI, but NOT CCF-ranked.** With CCF rank as the
criterion, the targets are ACL/AAAI/IJCAI (A) and EMNLP/NAACL/COLING (B).

---

## 3. CCF-aware fit (full-paper assumption: theorems + ablations included)

### CCF-A track

| Venue | Feasibility | Why |
|---|---|---|
| **ACL** | ★★★★☆ | Theory + formal isomorphism (P1–P5 as theorems) + cross-domain validation + downstream utility is a legitimate ACL contribution **iff** the theorem section is rigorous and empirical story complete (ablations + significance). MDI's strength is the theory once formalized — carries the paper, not SOTA. |
| **AAAI** | ★★★★☆ | Accepts formal AI theory + application; "AI & Law" is an established AAAI theme. Formal theorems carry weight. Annual (~Jan next, deadline ~mid previous year). |
| **IJCAI** | ★★★☆☆ | Similar to AAAI, biennial (next ~2027). Strong credit; schedule permitting, a good A-tier alternative. |

### CCF-B track

| Venue | Feasibility | Why |
|---|---|---|
| **EMNLP** | ★★★★★ | The most realistic CCF target. The full package (theorems + ablations + 3-layer utility + 8-dataset matrix) clears EMNLP's "solid + novel" bar comfortably. Annual (~Nov) — earliest viable CCF submission. |
| **NAACL / COLING** | ★★★☆☆ | Fine backups; NAACL spring (timing tight), COLING biennial (verify cycle). |

### Domain-alternate (only if CCF deprioritized)

| Venue | Fit | Why |
|---|---|---|
| **NLLP @ EMNLP** | ★★★★★ | Right audience for the isomorphism claim; no CCF rank. |
| **ICAIL** | ★★★★☆ | Formal-theory venue; ~2027 cycle; no CCF rank. |
| **JURIX** | ★★★★☆ | Legal-KB/representation; no CCF rank. |

---

## 4. Recommendation (CCF-aware)

**Primary: EMNLP (CCF B)** — realistic, annual, earliest CCF submission; the
complete MDI package (theory + representation + cross-domain + ablation +
downstream) fits its scope with the highest acceptance probability among
CCF-ranked venues.

**Stretch: ACL (CCF A)** — feasible once theorem formalization + ablation suite
are done; the theory (not SOTA) is the selling point, so the formal section
must carry it.

**Alternate A-tier: AAAI** — if scheduling favors a January deadline over
EMNLP's November cycle, or the paper reads more "AI theory" than "NLP".

**Domain-alternate (optional):** NLLP / ICAIL / JURIX remain excellent *if* CCF
rank is ever deprioritized — they are the most receptive audience for the
isomorphism claim.

---

## 5. Prerequisites before any CCF-A/B submission (all planned)

1. **Theorem formalization** (P1–P5 as provable statements + proof sketches) —
   turns "empirical regularity" into "theory", the core asset for ACL/AAAI.
2. **Ablation suite** — isolate contributions of (a) alignment supervision,
   (b) projection rank, (c) base feature choice, (d) feature augmentation vs.
   pure φ. Required by A/B reviewers.
3. **Related-work novelty confirmation** — formal search (ACL Anthology,
   Scholar) for prior "mathematical-doctrinal isomorphism"; hard prerequisite.
4. **Statistical significance** on the headline L3 gain (e.g., legalbert+phi on
   CUAD/ContractNLI with CIs) — converts "consistent small gains" into a
   defensible empirical claim.

---

## 6. Timeline (verify on official CFPs)

- EMNLP ~annual Nov (submission window ~May–Jul in ARR / direct).
- ACL ~annual Jul (ARR rolling; target a cycle ~6–7 months out).
- AAAI ~annual Jan (deadline ~mid previous year).
- ICAIL/JURIX/NLLP: domain-specialist calendars — only if CCF deprioritized.

---

*Rev. 2 corrects: (1) CCF dimension added per user; (2) theorem formalization
and ablation treated as **completed** (not missing); (3) NLLP/ICAIL/JURIX moved
from primary to domain-alternate because they are not CCF-ranked. CCF ranks per
CCF 2022 recommended list — verify officially before scheduling.*
