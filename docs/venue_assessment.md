# Venue Assessment — MDI (Mathematical-Doctrinal Isomorphism)

**Date:** 2026-08-13 (rev. 3 — full-paper, current evidence)
**Scope:** **full-paper assumption** — theorem formalization (P1–P5) and
ablation studies are treated as **completed** (user: both will be done; do not
rate by current in-progress state). Assessment dimension = **CCF recommended
venues** (user requirement).
**Version:** 0.2.6 evidence base + planned theorem/ablation.

---

## 0. Corrections

- Rev. 1: missed CCF dimension; treated theorem/ablation as missing. Fixed.
- Rev. 3: rating is by the **complete paper** (theorem + ablation + channel
  mechanism included), NOT by the current in-progress state — per user.

---

## 1. Evidence profile (what the COMPLETE paper sells)

| Asset | Strength (complete paper) | Honest caveat |
|---|---|---|
| Paradigm novelty | **数理法理同构 paradigm** — mapping + channel, two standard steps; dual-space Φ + P1–P5 | Novelty claim must be confirmed by related-work search (required) |
| Theory | **P1–P5 formalized as theorems** + axioms loaded into training (P2/P3/P5 in objective) | Theorem proofs must be complete before submission |
| Method (MDI-φ v2b) | Linear low-rank (768→64), interpretable, satisfies P2/P3/P5 (P3 full E<N<C, ‖W‖₂=2.0) | Not a representation-learning SOTA; 64-d projection |
| Mapping quality | Isometry leader: ContractNLI **0.215 (d=1.18)**, WillsNLI 0.338, ECHR 0.462 (single-dataset protocol) | SARA boundary (all n.s.) |
| Channel mechanism | Fit-and-select works: SCOTUS/CUAD→rbf, LEDGAR/MAUD→linear (channel diversity); dist channel AUC 0.187 on ContractNLI | φ channels below 768-d raw mpnet in absolute terms |
| Ablation (planned) | Each axiom's contribution isolated (alignment supervision / rank / base / augmentation) | Must be run |
| Applications | APPL-1 map-retrieval 13×, APPL-2 diff-vector 0.615, APPL-3 translation | Applications are Channel-step evidence, not SOTA |

**Net positioning (complete paper):** a *formalized paradigm (mapping + channel)
+ interpretable representation + cross-domain consistency + ablation-supported
+ channel-selection mechanism*. The selling point is the paradigm, not SOTA.

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

## 4. Recommendation (CCF-aware, complete-paper basis)

**Tier of the complete paper:** a formalized paradigm (mapping + channel) with
theorem-supported P1–P5, full ablation, cross-domain evidence (isometry leader
d=1.18), and a working channel-selection mechanism. **This is a CCF-A-stretch
paper** — it is NOT the "realistic target is B" profile.

**Primary: ACL (CCF A)** — the complete paper (formalized paradigm + theorems +
ablation + cross-domain + channel mechanism) is a legitimate ACL theory
contribution; the formal section (P1–P5 theorems) carries it, not SOTA. This is
the appropriate ambition once theorems + ablation are done.

**Co-primary: AAAI (CCF A)** — accepts formal AI theory + application; "AI &
Law" theme; formal theorems carry weight. Choose ACL vs AAAI by narrative
(NLP-flavored → ACL; AI-theory-flavored → AAAI) and by deadline.

**Backup: EMNLP (CCF B)** — the complete package clears EMNLP comfortably; a
safe annual fallback if A-track timing misses.

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
