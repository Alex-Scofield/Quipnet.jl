# Inverse method

This note describes the discrete inverse solver implemented in `Quipnet`.
The implementation supports any dimension `D`; the 1D, 2D, and 3D paths share
the same types and functions.

## Forward model

`SplitStepSchrodinger` evolves a state under

$$
i\,\partial_t \psi = -\tfrac12 \nabla^2 \psi + V_\theta \psi
$$

on a periodic Cartesian grid. For a time interval `T`, `_nsteps` chooses
`n` steps and `_splitstep_factors` builds

$$
\psi_{j+1} = K V_j \psi_j,
\qquad
K = \mathcal F^{-1}\operatorname{diag}\!\left(e^{-i\Delta t\,|k|^2/2}\right)\mathcal F,
\qquad
V_j = \operatorname{diag}\!\left(e^{-i\Delta t\,V_\theta}\right),
$$

with `Δt = T/n`, `Kfac` representing `K`, and `Vfac` representing `V_j`.
`_step_forward` applies one step, `schrod_periodic` applies `n` steps, and
`_propagate_with_history` stores the trajectory used by the adjoint.
`propagate` and `apply_forward` expose the final-state map.

The initial state is produced by `FourierInput` (or `BatchInput`) via
`apply_input`:

$$
\psi_0(x) =
\sum_{k_1=-m_1}^{m_1}\cdots\sum_{k_D=-m_D}^{m_D} w_k
\prod_{d=1}^{D} e^{i 2\pi k_d x_d/L_d},
$$

normalized in the grid's volume measure. Coefficient order is first-axis
fastest. `PointDensity` is the measurement map

$$
B(\psi) = \operatorname{vec}\!\left(|\psi|^2[I]\right),
$$

where `I` is `indices`, which may be linear indices or `CartesianIndex`
values (`Colon()` means the whole grid); a strict subset gives sparse point
measurements. `apply_measurement` concatenates measurements from a tuple of
states.

## Potential bases

`FourierBasis{D}` uses the tensor-product real Fourier basis. On each axis,

$$
e_{d,0}(x_d)=1,\qquad
e_{d,2k-1}(x_d)=\cos\!\left(\frac{2\pi k x_d}{L_d}\right),\qquad
e_{d,2k}(x_d)=\sin\!\left(\frac{2\pi k x_d}{L_d}\right),
$$

for `k = 1…m_d`, with `m_d = modes[d]`. The global constant is removed, so

$$
V_\theta(x) = \sum_{a\neq 0} \theta_a \prod_{d=1}^{D} e_{d,a_d}(x_d),
\qquad
\dim(b) = \prod_{d=1}^{D}\left(1+2m_d\right)-1,
$$

where `a` is the multi-index of basis functions and `a = 0` is the excluded
constant. `basis_matrix(b, grids)` returns the real matrix `Φ` whose columns
are the basis functions evaluated on the grid; columns are ordered first-axis
fastest. `evaluate_basis` computes `Φθ` and reshapes it to the grid. In `D`
dimensions `Φ` has size `prod(length.(grids)) × dim(b)`, and `loss_and_grad`
builds it once before the trajectory loop.

`GridBasis{D}` parameterizes the potential by its values at the solver grid
points. The coefficients are `θ = vec(V)`, ordered first-axis fastest, so

$$
\dim(b) = \prod_{d=1}^{D} N_d,
\qquad \Phi = I,
$$

and `evaluate_basis(b, θ, grids)` simply reshapes `θ` to the grid. The adjoint
of the identity basis is pointwise:

$$
\nabla_\theta J = 2\operatorname{Re}\!\left(\operatorname{vec}(c)\right),
$$

so no extra interpolation or matrix product is needed. `GridBasis(grids)`
stores the grid shape and requires it to match the forward solver's grid; a
mismatch raises an `ArgumentError`. Off-grid point evaluation is not
supported. Unlike `FourierBasis`, the grid basis contains the global constant:
density data cannot identify it, so with `ε > 0` the L2 term selects a mean,
and `relative_error` removes it by default. With `ε = 0` the constant is a
null direction of the objective.

## Inverse problem

`InverseProblem` stores the forward solver, input map, measurement map, a
potential basis (`FourierBasis` or `GridBasis`), `GaussianNoise`, input
coefficients `w_i`, observation times, and measurement vectors.
`generate_data` fills `measurements[i]` by simulating the forward map; `σw`
perturbs the inputs if nonzero, and `σμ` adds measurement noise if nonzero.

For `m = length(problem.inputs)`, `n_t = length(problem.times)`, and
`σ² = problem.noise.σμ^2`, `loss_and_grad` minimizes

$$
J(\theta) =
\frac{1}{m}\sum_{i=1}^{m}
\frac{1}{\sigma^2}
\left\|B\!\left(F_{V_\theta}(w_i)\right)-\mu_i\right\|^2
+ \varepsilon R(\theta),
$$

where `F_{V_θ}(w_i)` is the forward map at all `n_t` observation times, `B`
concatenates those final states, and `μ_i` is `measurements[i]`. The default
`L2Regularizer` has
`R(θ) = ||θ||²` and `∇R(θ) = 2θ`. The residual in the code is
`r = μhat - μobs`.

## Discrete adjoint

The gradient differentiates the discrete split-step map, not the continuous
Schrödinger equation. Let `ψ_j` be a stored trajectory, with final state
`ψ_n`, and let `ρ` be the residual block for one final state (`rt` in
`loss_and_grad`). `measurement_adjoint` initializes

$$
\lambda_n = \frac{2}{\sigma^2}\,\rho \odot \psi_n
$$

for `PointDensity`. For `j = n, n-1, …, 1`, the adjoint recursion is

$$
b_j = K^{\mathsf H}\lambda_j,
\qquad
c_j = \operatorname{conj}(b_j)\odot
\left(-i\,\Delta t\,V_j\psi_{j-1}\right),
$$

$$
\nabla_\theta J \mathrel{+}= 2\operatorname{Re}\!\left(\Phi^{\mathsf T}c_j\right),
\qquad
\lambda_{j-1} = V_j^{\mathsf H} b_j.
$$

In `loss_and_grad` these are implemented as
`b = ifft(conj.(Kfac) .* fft(λ))`,
`c = conj.(b) .* (-im .* dt .* (Vfac .* ψj))`, and
`transpose(Φ) * vec(c)`. The per-experiment gradient is divided by `m`, then
the regularizer gradient is added.

## Optimization and gauge

`TikhonovSolver` wraps `Optim.LBFGS` and passes `loss_and_grad` through
`Optim.NLSolversBase.only_fg!`. `fit!` starts from
`θ0 = zeros(dim(problem.basis))`, calls the optimizer, and returns an
`InverseResult` containing `θ`, `V = evaluate_potential(problem, θ)`,
the objective, iteration count, and convergence flag. `reconstruct(result)`
returns the potential.

An additive constant in the potential is not identifiable from `PointDensity`
data. `FourierBasis` excludes it by construction; `GridBasis` includes it and
relies on the regularizer (for `ε > 0`) to select a mean. `relative_error`
therefore removes each potential's mean by default
(`modulo_constant=true`).

## Practical notes

- The analytic adjoint is implemented only for `SplitStepSchrodinger`;
  `FreeSchrodinger` has no potential and is rejected by `loss_and_grad`.
- `PointDensity` produces a dense measurement vector. The stored trajectories
  also scale with the full grid, so large 3D problems require modest `N` or a
  matrix-free/checkpointed implementation. `FourierBasis` builds a dense `Φ`
  once per `loss_and_grad` call; `GridBasis` uses a diagonal identity (`O(N)`
  memory) and does not need an interpolation matrix.
- The discrete adjoint uses the same `Kfac` and `Vfac` as the forward step, so
  it is the exact gradient of the discrete objective; finite differences agree
  up to truncation and floating-point error.

See `examples/sparse_measurements.jl` and
`examples/sparse_measurements_triangle.jl` for full-grid versus sparse-sensor
comparisons; they write `inverse_sparse_demo.png` and
`inverse_sparse_triangle_demo.png`.
