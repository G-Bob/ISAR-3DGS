# Geometry-Only Encounter Selection and Minimum-Count Inference

## 1. Boundary of the claim

The object structure is unknown before reconstruction. Therefore encounter
selection must not use an ISAR image to infer object-surface coverage, and it
must not use the reference GLB. The selector reads only acquisition geometry:

\[
\texttt{view\_angle}[v,0{:}3]=\mathbf k_{r,v},\qquad
\texttt{view\_angle}[v,3{:}6]=\mathbf k_{d,v}.
\]

Here \(\mathbf k_r\) is the target-to-radar direction and \(\mathbf k_d\) is
the cross-range direction. Every encounter contributes the same number of
images sampled over the same relative aperture, so encounters have equal cost.

The resulting percentage is **observation-direction coverage**, not unknown
object-surface area coverage.

## 2. Directional coverage

The selector places equal-area samples on a Fibonacci sphere. A sample is
covered when it is within `direction_cap_deg` of at least one selected
\(\mathbf k_r\). Union coverage favors encounter sets that span new observation
directions rather than repeat existing ones.

This is a target-independent acquisition-space proxy. It does not assert which
parts of an unknown object are visible, shadowed, or reflective.

## 3. Projection observability

Each image supplies range and cross-range projection directions:

\[
H_v=\begin{bmatrix}\mathbf k_{r,v}^{\mathsf T}\\
                    \mathbf k_{d,v}^{\mathsf T}\end{bmatrix},\qquad
F(S)=\frac{1}{|S|M}\sum_{e\in S}\sum_{v=1}^{M}H_v^{\mathsf T}H_v.
\]

The normalized smallest eigenvalue
\(\lambda_{\min}(F)/\operatorname{tr}(F)\) measures how well the weakest 3-D
direction is constrained. The selector also reports eigenvalue isotropy and
the information condition number. These metrics depend only on camera/radar
geometry.

## 4. Independent fixed-K optimization

For every requested encounter count K, the program solves

\[
S_K^\star=\arg\max_{S:\,|S|=K}J(S),
\]

with

\[
J(S)=w_d C_{\mathrm{dir}}(S)
     +w_o\frac{\lambda_{\min}(F(S))}{\operatorname{tr}(F(S))}
     +w_b\frac{\overline{\Delta\theta}(S)}{180^\circ},
\]

where \(\overline{\Delta\theta}\) is mean separation between encounter center
directions. Different K values are optimized independently; in general
\(S_K^\star\) is not a subset of \(S_{K+1}^\star\). If
\(\binom{N}{K}\) is below `max_exact_combinations`, every fixed-K subset is
evaluated. Larger spaces use multiple deterministic random starts followed by
best-improvement one-encounter swap search. Each K still has an independent
solution, but large-search results are approximate and must not be described
as guaranteed global optima. `fixed_k_selection.csv` records the method used.

## 5. Training and evaluation path

1. Verify equal frame count and equal relative aperture for all encounters.
2. Run geometry-only selection and freeze the independently optimized subset
   for every tested K.
3. Train each tested K using exactly the same images per encounter, optimizer,
   update count, capacity, seed and loss settings.
4. Evaluate every K on the same frozen test-frame list.
5. Evaluate each PLY independently at `q >= 0.1`, `q >= 0.3` and `q >= 0.5`.
6. Use `../3dmodel/manipulator_scale1p8.glb` only at this post-training stage to
   calculate Chamfer distance, precision, recall and F-score.
7. Repeat the candidate K-1, K and K+1 with at least three seeds.

The default automatic decision at `q=0.3` selects the smallest K that reaches
the configured fractions of full-set direction coverage, full-set projection
observability, best geometric score, and acceptable connected-component
quality. These ratios are experiment rules, not physical constants.

## 6. Unknown-target inference

Without any reference geometry, absolute geometric correctness cannot be
proven from the acquisition alone. Report the minimum K as a
**self-consistency minimum**, using held-out reprojection quality, stability
from K to K+1, agreement across q thresholds, point-cloud compactness and seed
repeatability. A ground-truth-backed minimum and a self-consistency minimum are
different claims and should be labeled separately.
