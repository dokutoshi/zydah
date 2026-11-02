# Zadeh's Paradox, Yager, and Smets: A Derivation from First Principles

Nov 1, 2026 · @Leo P.

## 0. Purpose and notation

Two sources that each give a hypothesis 1% support can be fused by Dempster's rule into 100% certainty in that hypothesis. This document builds the theory from the frame of discernment up, derives that result exactly, shows why it is a genuine instability rather than a curiosity, and then shows how Yager (1987) and Smets (1990) each repair it. Every quantity is computed in full; nothing is assumed beyond the definitions given.

Notation is fixed once here and used unchanged throughout.

| Symbol | Meaning |
| --- | --- |
| Θ | Frame of discernment: the set of candidate answers |
| M, T, C | Hypotheses (elements of Θ): Meningitis, Tumor, Concussion |
| X, Y, A | Subsets of Θ (sets of hypotheses). Never used for a hypothesis |
| {T} | The subset containing only T. Masses are always written on sets: m({T}), never m(T) |
| ∅ | The empty set |
| Ā | Complement of A in Θ |
| \|X\| | Number of elements in X |
| m₁, m₂ | Mass functions of source 1 and source 2 (subscript = source) |
| m∩, m\_D, m\_Y, m\_S | Combined mass under: unnormalized conjunctive rule, Dempster, Yager, Smets' TBM (subscript = rule) |
| K | Conflict: the combined mass that lands on ∅ |
| ε, δ | Small positive numbers, 0 < ε, δ < 1 |
| ω | A generic single hypothesis (any one of M, T, C) |

Formulas are displayed on their own lines; in running text the same symbols appear as plain characters.

## 1. The frame of discernment

The frame Θ is the list of answers we are choosing among. In Zadeh's example a patient has exactly one of three conditions:

```latex
\Theta = \{M,\ T,\ C\}
```

Two assumptions are built into a frame, and both matter later:

1. **Exclusive.** At most one hypothesis is true. The patient cannot have both M and T.
2. **Exhaustive (closed world).** At least one hypothesis is true. The answer is somewhere in Θ.

Together they say: exactly one element of Θ is the truth. Section 9 drops assumption 2.

Evidence is often about *groups* of answers rather than single answers. A test that rules out meningitis supports "T or C" without choosing between them. So evidence is attached to subsets of Θ. The collection of all subsets is the power set 2^Θ. With 3 hypotheses it has 2³ = 8 members:

```latex
2^{\Theta} = \{\ \emptyset,\ \{M\},\ \{T\},\ \{C\},\ \{M,T\},\ \{M,C\},\ \{T,C\},\ \Theta\ \}
```

How to read a subset: putting weight on {T, C} means "the truth is T or C, and this evidence cannot tell which." Putting weight on Θ means "the truth is somewhere in the frame", which says nothing. Putting weight on ∅ means "the truth is nowhere in the frame", which contradicts assumption 2.

## 2. Mass functions

A source of evidence is described by a mass function m, which distributes one unit of evidential weight over the subsets of Θ:

```latex
m : 2^{\Theta} \to [0,1], \qquad m(\emptyset) = 0, \qquad \sum_{X \subseteq \Theta} m(X) = 1
```

The condition m(∅) = 0 is the closed-world assumption from Section 1 written as an equation. The subsets X with m(X) > 0 are the source's **focal elements**.

The key difference from probability: mass placed on a set is *not* divided among its members. If m({T, C}) = 0.3, that 0.3 says "T or C" and nothing more. It is not 0.15 each.

Three examples on Θ = {M, T, C}:

| Name | Focal elements and masses | What the source says |
| --- | --- | --- |
| Vacuous | m(Θ) = 1 | Nothing at all. Total ignorance |
| Bayesian | m({M}) = 0.99, m({T}) = 0.01 | Every focal element is a single hypothesis. This is an ordinary probability distribution |
| Partial | m({T}) = 0.6, m({T,C}) = 0.3, m(Θ) = 0.1 | Strong support for T, some weaker support for "T or C", a little uncommitted |

Check each sums to 1: 1; 0.99 + 0.01 = 1; 0.6 + 0.3 + 0.1 = 1.

Both sources in Zadeh's example are Bayesian. Section 11 shows that this choice is what makes the example extreme.

## 3. Belief and plausibility

From a mass function we read two numbers for any set A. **Belief** counts the mass that *must* be in A. **Plausibility** counts the mass that *could* be in A.

```latex
\mathrm{Bel}(A) = \sum_{X \subseteq A} m(X) \qquad\qquad \mathrm{Pl}(A) = \sum_{X \cap A \neq \emptyset} m(X)
```

Bel(A) adds every focal element lying entirely inside A. Pl(A) adds every focal element that touches A at all. Since anything inside A also touches A, Bel(A) ≤ Pl(A). The pair \[Bel(A), Pl(A)\] is the **belief interval**; its width measures ignorance about A.

**Duality.** When m(∅) = 0, a focal element either touches A or lies entirely inside Ā, never both and never neither. So:

```latex
\mathrm{Pl}(A) = 1 - \mathrm{Bel}(\bar{A}) \qquad \text{(valid only when } m(\emptyset) = 0\text{)}
```

This condition is easy to forget. Section 9 shows what goes wrong when it does not hold.

**Worked example** with the Partial source m({T}) = 0.6, m({T,C}) = 0.3, m(Θ) = 0.1:

| Set A | Focal elements inside A | Bel(A) | Focal elements touching A | Pl(A) | Duality check: 1 − Bel(Ā) |
| --- | --- | --- | --- | --- | --- |
| {T} | {T} | 0.6 | {T}, {T,C}, Θ | 1.0 | 1 − Bel({M,C}) = 1 − 0 = 1.0 |
| {C} | none | 0 | {T,C}, Θ | 0.4 | 1 − Bel({M,T}) = 1 − 0.6 = 0.4 |
| {M} | none | 0 | Θ | 0.1 | 1 − Bel({T,C}) = 1 − 0.9 = 0.1 |
| {T,C} | {T}, {T,C} | 0.9 | {T}, {T,C}, Θ | 1.0 | 1 − Bel({M}) = 1 − 0 = 1.0 |

Two boundary cases fix intuition:

- **Bayesian source:** every focal element is a singleton, so for any single hypothesis ω, Bel({ω}) = Pl({ω}) = m({ω}). The interval has zero width; belief equals probability.
- **Vacuous source:** every set except ∅ and Θ gets Bel = 0 and Pl = 1. The interval \[0, 1\] is complete ignorance.

## 4. Combining two sources: the conjunctive rule and conflict K

Suppose source 1 says "the truth is in X" and source 2 says "the truth is in Y". If both are right, the truth is in X ∩ Y. If the sources are independent, the weight on that joint claim is the product m₁(X)·m₂(Y). Collecting every pair that lands on the same set A gives the **unnormalized conjunctive combination**:

```latex
m_{\cap}(A) = \sum_{X \cap Y = A} m_1(X)\, m_2(Y) \qquad \text{for every } A \subseteq \Theta,\ \text{including } A = \emptyset
```

The products over all pairs sum to (Σ m₁)(Σ m₂) = 1 × 1 = 1, so no mass is created or lost. But some pairs may have X ∩ Y = ∅: the sources point to sets with nothing in common. That mass is the **conflict**:

```latex
K = m_{\cap}(\emptyset) = \sum_{X \cap Y = \emptyset} m_1(X)\, m_2(Y)
```

K = 0 means the sources never contradict each other. K = 1 means every pairing contradicts. The procedure is always the same: list each focal element of m₁ against each focal element of m₂, intersect, multiply.

**Example 4a: no conflict.** m₁({T}) = 0.6, m₁(Θ) = 0.4; m₂({T,C}) = 0.5, m₂(Θ) = 0.5.

| m₁ focal (mass) | m₂ focal (mass) | Intersection | Product |
| --- | --- | --- | --- |
| {T} (0.6) | {T,C} (0.5) | {T} | 0.30 |
| {T} (0.6) | Θ (0.5) | {T} | 0.30 |
| Θ (0.4) | {T,C} (0.5) | {T,C} | 0.20 |
| Θ (0.4) | Θ (0.5) | Θ | 0.20 |

Result: m∩({T}) = 0.60, m∩({T,C}) = 0.20, m∩(Θ) = 0.20, and K = 0. Total 1.00.

**Example 4b: partial conflict.** m₁({M}) = 0.6, m₁(Θ) = 0.4; m₂({T}) = 0.5, m₂(Θ) = 0.5.

| m₁ focal (mass) | m₂ focal (mass) | Intersection | Product |
| --- | --- | --- | --- |
| {M} (0.6) | {T} (0.5) | ∅ | 0.30 |
| {M} (0.6) | Θ (0.5) | {M} | 0.30 |
| Θ (0.4) | {T} (0.5) | {T} | 0.20 |
| Θ (0.4) | Θ (0.5) | Θ | 0.20 |

Result: m∩({M}) = 0.30, m∩({T}) = 0.20, m∩(Θ) = 0.20, and K = m∩(∅) = 0.30. Total 1.00.

m∩ is not yet a valid mass function under the closed world whenever K > 0, because it puts mass on ∅. Sections 5, 8 and 9 are three different answers to the question: *what do we do with K?*

## 5. Dempster's rule: delete the conflict and rescale

Dempster's answer is to discard the mass on ∅ and scale everything else up so it sums to 1 again:

```latex
m_D(A) = \frac{m_{\cap}(A)}{1 - K} \quad \text{for } A \neq \emptyset, \qquad m_D(\emptyset) = 0, \qquad \text{undefined if } K = 1
```

The reasoning: the closed world says the truth is in Θ, so the conflicting pairings "cannot have happened". We condition on the event that they did not. Dividing by 1 − K is exactly that conditioning.

**Example 4b under Dempster.** K = 0.30, so 1 − K = 0.70:

| Set | m∩ | m\_D = m∩ / 0.70 |
| --- | --- | --- |
| {M} | 0.30 | 0.4286 |
| {T} | 0.20 | 0.2857 |
| Θ | 0.20 | 0.2857 |
| ∅ | 0.30 | 0 (removed) |

Check: 0.4286 + 0.2857 + 0.2857 = 1.0000. Here the rescaling is mild and the result is reasonable.

The important feature is what the division does to the *size* of K: it disappears from the result. A combination with K = 0.30 and one with K = 0.9999 can return identical masses. Section 6 exploits this.

## 6. Zadeh's example

Two doctors examine the same patient. Doctor 1 is almost sure it is meningitis; doctor 2 is almost sure it is concussion. Each gives tumor a small residual weight.

**Claim.** For any 0 < ε < 1 and 0 < δ < 1, let

```latex
m_1(\{M\}) = 1-\varepsilon, \quad m_1(\{T\}) = \varepsilon, \qquad m_2(\{C\}) = 1-\delta, \quad m_2(\{T\}) = \delta
```

with all other masses zero. Then m\_D({T}) = 1, whatever the values of ε and δ.

**Step 1: list every pairing.** Each source has two focal elements, so there are 2 × 2 = 4 pairings:

| m₁ focal (mass) | m₂ focal (mass) | Intersection | Product |
| --- | --- | --- | --- |
| {M} (1 − ε) | {C} (1 − δ) | ∅ | (1 − ε)(1 − δ) |
| {M} (1 − ε) | {T} (δ) | ∅ | (1 − ε)δ |
| {T} (ε) | {C} (1 − δ) | ∅ | ε(1 − δ) |
| {T} (ε) | {T} (δ) | {T} | εδ |

Three of the four pairings are contradictory. Only the T–T pairing survives.

**Step 2: compute K.** Add the three ∅ rows and expand each product:

```latex
\begin{aligned}
K &= (1-\varepsilon)(1-\delta) + (1-\varepsilon)\delta + \varepsilon(1-\delta) \\
  &= (1 - \varepsilon - \delta + \varepsilon\delta) + (\delta - \varepsilon\delta) + (\varepsilon - \varepsilon\delta) \\
  &= 1 - \varepsilon\delta
\end{aligned}
```

In the second line, −ε cancels +ε, −δ cancels +δ, and the εδ terms give +εδ − εδ − εδ = −εδ.

**Step 3: apply Dempster's rule.** The only non-empty result is m∩({T}) = εδ, and 1 − K = εδ:

```latex
m_D(\{T\}) = \frac{m_{\cap}(\{T\})}{1-K} = \frac{\varepsilon\delta}{\varepsilon\delta} = 1, \qquad m_D(\{M\}) = m_D(\{C\}) = 0 \qquad \blacksquare
```

The numerator and denominator are the same quantity, so ε and δ cancel completely. That is why the result does not depend on how small they are.

**The numerical instance, ε = δ = 0.01.**

| Pairing | Product | Intersection |
| --- | --- | --- |
| {M} × {C} | 0.99 × 0.99 = 0.9801 | ∅ |
| {M} × {T} | 0.99 × 0.01 = 0.0099 | ∅ |
| {T} × {C} | 0.01 × 0.99 = 0.0099 | ∅ |
| {T} × {T} | 0.01 × 0.01 = 0.0001 | {T} |

Check: 0.9801 + 0.0099 + 0.0099 + 0.0001 = 1.0000.

So K = 0.9999, 1 − K = 0.0001, and m\_D({T}) = 0.0001 / 0.0001 = 1.

**Why this is disturbing.** Both sources are Bayesian, so their belief intervals for T have zero width: Bel₁({T}) = Pl₁({T}) = 0.01 and Bel₂({T}) = Pl₂({T}) = 0.01. Each doctor considers tumor the least likely diagnosis. After fusion, Bel\_D({T}) = Pl\_D({T}) = 1: tumor is certain, and the two diagnoses each doctor actually favored have been eliminated. A 99.99% conflict was removed by the division and left no trace in the output.

## 7. Why it is pathological: the rule is discontinuous at total conflict

The 0.01 → 1 amplification is striking, but the rigorous objection is stronger: near total conflict, Dempster's output is decided by perturbations too small to matter. To prove this it is not enough to show one sequence of inputs whose output stays at 1; that sequence has a perfectly good limit. We need **two sequences that converge to the same inputs but produce different outputs.**

**The limit point.** Take the fully confident pair m₁\*({M}) = 1 and m₂\*({C}) = 1. Their only pairing is {M} ∩ {C} = ∅, so K = 1 and Dempster's rule is undefined there.

**Path A (Zadeh's path).** Leak a small amount ε of each source onto T:

- m₁({M}) = 1 − ε, m₁({T}) = ε
- m₂({C}) = 1 − ε, m₂({T}) = ε

Section 6 with δ = ε gives m\_D({T}) = 1 for every ε > 0.

**Path B.** Leak ε of each source onto the *other* source's favorite:

- m₁({M}) = 1 − ε, m₁({C}) = ε
- m₂({C}) = 1 − ε, m₂({M}) = ε

| m₁ focal (mass) | m₂ focal (mass) | Intersection | Product |
| --- | --- | --- | --- |
| {M} (1 − ε) | {C} (1 − ε) | ∅ | (1 − ε)² |
| {M} (1 − ε) | {M} (ε) | {M} | ε(1 − ε) |
| {C} (ε) | {C} (1 − ε) | {C} | ε(1 − ε) |
| {C} (ε) | {M} (ε) | ∅ | ε² |

```latex
\begin{aligned}
K &= (1-\varepsilon)^2 + \varepsilon^2 = 1 - 2\varepsilon + 2\varepsilon^2 \\
1 - K &= 2\varepsilon - 2\varepsilon^2 = 2\varepsilon(1-\varepsilon) \\
m_D(\{M\}) &= \frac{\varepsilon(1-\varepsilon)}{2\varepsilon(1-\varepsilon)} = \tfrac{1}{2}, \qquad m_D(\{C\}) = \tfrac{1}{2}
\end{aligned}
```

Numerical check at ε = 0.01: products 0.9801, 0.0099, 0.0099, 0.0001; K = 0.9802; m\_D({M}) = m\_D({C}) = 0.0099 / 0.0198 = 0.5.

**Comparing the two paths.** Measure the distance between two mass functions by the largest difference in any single mass. On both paths each source is within ε of the limit point, so the inputs converge to the same place as ε → 0.

|  | Input distance from limit point | Dempster output (for every ε > 0) |
| --- | --- | --- |
| Path A | ε for each source | m\_D({T}) = 1 |
| Path B | ε for each source | m\_D({M}) = m\_D({C}) = 0.5 |

The two outputs differ by 1 in the mass on {T}, the largest possible difference, and this gap does not shrink as ε → 0. So no value assigned at the limit point can make the rule continuous there. Near total conflict, *where the last 1% of each source happens to sit* fully determines the answer. This is the precise content of Zadeh's objection.

## 8. Yager's rule (1987): conflict becomes ignorance

Yager keeps the closed world but refuses to pretend the conflict did not happen. Instead of deleting K and rescaling, he moves K onto Θ, the set that means "somewhere in the frame, we cannot say where":

```latex
m_Y(A) = m_{\cap}(A) \ \text{ for } \emptyset \neq A \neq \Theta, \qquad m_Y(\Theta) = m_{\cap}(\Theta) + K, \qquad m_Y(\emptyset) = 0
```

No division occurs, so nothing is amplified and the rule is defined even at K = 1. The interpretation: when sources contradict each other, the honest conclusion is that we know less, not more.

**Zadeh's numbers under Yager** (ε = δ = 0.01). From Section 6, m∩({T}) = 0.0001, m∩(Θ) = 0, K = 0.9999:

| Set | m∩ | m\_Y |
| --- | --- | --- |
| {T} | 0.0001 | 0.0001 |
| {M}, {C} | 0 | 0 |
| Θ | 0 | 0 + 0.9999 = 0.9999 |
| ∅ | 0.9999 | 0 (moved to Θ) |

Belief intervals, using Section 3 (valid here because m\_Y(∅) = 0):

- T: Bel\_Y({T}) = 0.0001; Pl\_Y({T}) = m\_Y({T}) + m\_Y(Θ) = 1.0
- M: Bel\_Y({M}) = 0; Pl\_Y({M}) = m\_Y(Θ) = 0.9999. The same holds for C.

Every hypothesis now has an interval close to \[0, 1\]: near-total ignorance, which is the defensible reading of two experts who flatly disagree.

**Path B under Yager** (ε = 0.01): m\_Y({M}) = 0.0099, m\_Y({C}) = 0.0099, m\_Y(Θ) = 0.9802. Paths A and B now differ by at most 0.0197 in any mass (on Θ: 0.9999 − 0.9802), and in general that gap is 2ε − 3ε², which goes to 0 with ε. Yager's rule removes the discontinuity of Section 7.

**A cost (literature).** Yager's rule is commutative but not associative: combining (m₁ with m₂) and then m₃ can differ from combining m₁ with (m₂ with m₃), because conflict from the first step is already sitting on Θ when the second step runs. With more than two sources, the standard remedy is to combine all sources in a single step, computing one conflict over all of them together.

## 9. Smets' Transferable Belief Model (1990, 1994): keep the conflict as a signal

Smets questions assumption 2 of Section 1. Perhaps the truth is not in Θ at all: the patient has a fourth condition nobody listed, or one of the doctors is simply wrong. This is the **open world**. In it, mass on ∅ is meaningful: it measures how strongly the evidence says "none of the listed answers fits".

The TBM separates two levels:

1. **Credal level (what we believe).** Combine with the unnormalized conjunctive rule and keep the result as is, including m\_S(∅) = K. No division, no transfer.
2. **Pignistic level (what we bet on).** Only when a decision is forced, convert beliefs into probabilities.

**Credal level.** With m(∅) allowed to be positive, belief must exclude ∅ explicitly, and duality changes:

```latex
\mathrm{Bel}_S(A) = \sum_{\emptyset \neq X \subseteq A} m_S(X), \qquad \mathrm{Pl}_S(A) = \sum_{X \cap A \neq \emptyset} m_S(X), \qquad \mathrm{Pl}_S(A) = 1 - m_S(\emptyset) - \mathrm{Bel}_S(\bar{A})
```

The closed-world identity Pl(A) = 1 − Bel(Ā) from Section 3 no longer holds, because mass on ∅ lies neither inside Ā nor touching A.

**Zadeh's numbers, credal level.** m\_S({T}) = 0.0001 and m\_S(∅) = 0.9999.

- Bel\_S({T}) = 0.0001
- Pl\_S({T}) = 0.0001. The ∅ mass does not intersect {T}, so it is not counted.
- Duality check: 1 − 0.9999 − Bel\_S({M,C}) = 0.0001 − 0 = 0.0001. Correct.

The belief interval for T is \[0.0001, 0.0001\], and 0.9999 of the evidence says the frame itself has failed. (Applying the closed-world identity here would wrongly give Pl\_S({T}) = 1.0.)

**Pignistic level.** Write ω for a single hypothesis. The pignistic probability splits each focal element's mass equally among its members, then rescales to exclude ∅:

```latex
\mathrm{BetP}(\omega) = \frac{1}{1 - m(\emptyset)} \sum_{X \ni \omega} \frac{m(X)}{|X|}, \qquad \text{defined when } m(\emptyset) < 1
```

Equal splitting is the principle of insufficient reason, applied only at the moment of decision. Two quick checks: a vacuous source gives BetP = 1/3 for each of M, T, C; a Bayesian source gives BetP = its own masses.

For Zadeh's numbers:

```latex
\mathrm{BetP}_S(T) = \frac{1}{1 - 0.9999} \cdot \frac{0.0001}{1} = \frac{0.0001}{0.0001} = 1
```

**The TBM decision always equals Dempster's.** Dempster divides every non-empty mass by 1 − K. BetP divides by 1 − m\_S(∅) = 1 − K. The two divisions are the same, so BetP of the TBM result equals BetP of the Dempster result for every input. Smets does not dispute the bet; he disputes destroying K before anyone looks at it. Read at the credal level, m\_S(∅) = 0.9999 is an alarm: the bet on T is conditional on the truth lying in Θ, and the evidence says that condition almost certainly fails. The correct action is to suspend the decision and revise the frame or check the sources, not to report tumor.

## 10. Side by side on Zadeh's numbers

All three rules start from the same m∩: {T} = 0.0001 and ∅ = K = 0.9999. They differ only in where K goes, and that single choice decides everything below it.

| Quantity | Dempster (1967) | Yager (1987) | Smets TBM (1990) |
| --- | --- | --- | --- |
| World assumption | Closed | Closed | Open |
| Where K = 0.9999 goes | Deleted by division | To Θ | Kept on ∅ |
| m({T}) | 1 | 0.0001 | 0.0001 |
| m({M}), m({C}) | 0 | 0 | 0 |
| m(Θ) | 0 | 0.9999 | 0 |
| m(∅) | 0 | 0 | 0.9999 |
| Bel({T}) | 1 | 0.0001 | 0.0001 |
| Pl({T}) | 1 | 1.0 | 0.0001 |
| Pl({M}), Pl({C}) | 0 | 0.9999 | 0 |
| BetP(T) | 1 | 0.0001 + 0.9999/3 = 0.3334 | 1 (conditional on truth in Θ) |
| BetP(M), BetP(C) | 0 | 0.9999/3 = 0.3333 | 0 |
| Continuous at K = 1? | No (Section 7) | Yes | Credal level yes; BetP no (it inherits Dempster's jump) |
| Reading | Tumor is certain | We know almost nothing | The frame or a source is wrong; do not decide |

Check the Yager BetP row: 0.3334 + 0.3333 + 0.3333 = 1.0000.

Dempster and the TBM agree on the bet but disagree on the credal state; Yager and the TBM agree that T has only 0.0001 of belief but disagree on whether the missing mass means "don't know" (Θ) or "none of these" (∅).

## 11. What the paradox does and does not show

The paradox shows that normalization erases the magnitude of conflict and makes Dempster's rule discontinuous at K = 1. It does not show that Dempster's rule is wrong in general. Haenni (2005) argues that the inputs are the deeper problem: each doctor is modeled as a Bayesian source with m(Θ) = 0, which means each is assumed to be perfectly reliable. Two perfectly reliable sources cannot disagree, so feeding the rule two that do is a modeling error.

**Reliability discounting** (Shafer 1976) makes the point concrete. If a source is reliable with probability α, scale its committed masses by α and move the rest to Θ:

```latex
m^{\alpha}(X) = \alpha\, m(X) \ \text{ for } X \neq \Theta, \qquad m^{\alpha}(\Theta) = 1 - \alpha + \alpha\, m(\Theta)
```

With α = 0.9 on Zadeh's sources: m₁ becomes {M} 0.891, {T} 0.009, Θ 0.1, and m₂ becomes {C} 0.891, {T} 0.009, Θ 0.1. There are now 3 × 3 = 9 pairings:

| m₁ focal | m₂ focal | Intersection | Product |
| --- | --- | --- | --- |
| {M} 0.891 | {C} 0.891 | ∅ | 0.793881 |
| {M} 0.891 | {T} 0.009 | ∅ | 0.008019 |
| {M} 0.891 | Θ 0.1 | {M} | 0.0891 |
| {T} 0.009 | {C} 0.891 | ∅ | 0.008019 |
| {T} 0.009 | {T} 0.009 | {T} | 0.000081 |
| {T} 0.009 | Θ 0.1 | {T} | 0.0009 |
| Θ 0.1 | {C} 0.891 | {C} | 0.0891 |
| Θ 0.1 | {T} 0.009 | {T} | 0.0009 |
| Θ 0.1 | Θ 0.1 | Θ | 0.01 |

Totals: K = 0.809919; m∩({M}) = 0.0891; m∩({C}) = 0.0891; m∩({T}) = 0.000081 + 0.0009 + 0.0009 = 0.001881; m∩(Θ) = 0.01. These sum to 1.000000.

Dempster's rule divides by 1 − K = 0.190081:

| Set | m\_D, fully reliable sources (Section 6) | m\_D, sources discounted to α = 0.9 |
| --- | --- | --- |
| {M} | 0 | 0.4687 |
| {C} | 0 | 0.4687 |
| {T} | 1 | 0.0099 |
| Θ | 0 | 0.0526 |

Values are rounded to four places (unrounded they sum to 1). Admitting just 10% unreliability brings T from certainty back to about 1%, and leaves M and C as the two live, unresolved candidates, which is what the doctors actually reported. K is still 0.81, which still deserves to be flagged.

**Summary.** The pathology shows up in the division by 1 − K, and it is triggered by fully committed sources. There are therefore two complementary remedies: model source unreliability explicitly (discounting puts mass on Θ before combination), or keep the conflict visible after combination (Yager puts it on Θ, the TBM keeps it on ∅). A robust system typically does both.

## 12. References

| Year | Author | Work | Role here |
| --- | --- | --- | --- |
| 1967 | A. P. Dempster | Upper and lower probabilities induced by a multivalued mapping. *Annals of Mathematical Statistics* 38(2), 325–339 | Combination rule (Section 5) |
| 1976 | G. Shafer | *A Mathematical Theory of Evidence*. Princeton University Press | Mass, belief, plausibility, discounting (Sections 2, 3, 11) |
| 1979 | L. A. Zadeh | [On the Validity of Dempster's Rule of Combination of Evidence](https://www2.eecs.berkeley.edu/Pubs/TechRpts/1979/ERL-m-79-24.pdf). Memo UCB/ERL M79/24, Electronics Research Laboratory, UC Berkeley | First statement of the objection |
| 1984 | L. A. Zadeh | [Review of *A Mathematical Theory of Evidence*](https://aaai.org/ojs/index.php/aimagazine/issue/view/52). *AI Magazine* 5(3), 81–83 | The widely cited form of the paradox (Section 6) |
| 1986 | L. A. Zadeh | [A simple view of the Dempster-Shafer theory of evidence and its implication for the rule of combination](https://doi.org/10.1609/aimag.v7i2.542). *AI Magazine* 7(2), 85–90 | Follow-up; relational-database view |
| 1987 | R. R. Yager | On the Dempster-Shafer framework and new combination rules. *Information Sciences* 41(2), 93–137 | Conflict to Θ (Section 8) |
| 1990 | P. Smets | The combination of evidence in the transferable belief model. *IEEE Transactions on Pattern Analysis and Machine Intelligence* 12(5), 447–458 | Open world, m(∅) kept (Section 9) |
| 1994 | P. Smets, R. Kennes | The transferable belief model. *Artificial Intelligence* 66(2), 191–234 | Credal/pignistic levels, BetP (Section 9) |
| 2005 | R. Haenni | Shedding new light on Zadeh's criticism of Dempster's rule of combination. *Proceedings of FUSION 2005* | Input-modeling critique (Section 11) |

Linked entries were checked against the source pages; the others are standard bibliographic records cited from memory. All numerical results in this document were computed here from the stated definitions.
