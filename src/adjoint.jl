"""
    loss_and_grad(problem, θ; ε=0.0, regularizer=L2Regularizer())

Return the regularized objective and its gradient with respect to the
potential coefficients `θ`:

    J(θ) = (1/m) Σ_i ‖B(F_{V_θ}(w_i)) − μ_i‖² / σ² + ε R(θ)

The gradient is the discrete adjoint of the split-step propagator. The stored
trajectory is simulated, not measured; `B` is applied once per trajectory.
"""
function loss_and_grad(problem::InverseProblem, θ::AbstractVector;
        ε::Real = 0.0,
        regularizer::AbstractRegularizer = L2Regularizer())
    problem.forward isa SplitStepSchrodinger || throw(ArgumentError(
        "the analytic adjoint is implemented for SplitStepSchrodinger"))
    ε ≥ 0 || throw(ArgumentError("ε must be ≥ 0"))

    K = dim(problem.basis)
    length(θ) == K || throw(ArgumentError(
        "expected $K potential coefficients, got $(length(θ))"))

    grids = forward_grids(problem.forward)
    Φ = basis_matrix(problem.basis, grids)
    V = reshape(Φ * θ, map(length, grids))

    σ² = problem.noise.σμ^2
    m = length(problem.inputs)
    J = 0.0
    g = zeros(Float64, K)

    for i in 1:m
        μobs = problem.measurements[i]
        isempty(μobs) && throw(ArgumentError(
            "measurements[$i] is empty; call generate_data first"))

        ψ0s = apply_input(problem.input, problem.inputs[i], grids)
        ψ0s = ψ0s isa Tuple ? ψ0s : (ψ0s,)
        length(ψ0s) == length(problem.times) || throw(ArgumentError(
            "input map produced $(length(ψ0s)) states for $(length(problem.times)) times"))

        trajs = ntuple(t -> _propagate_with_history(problem.forward,
                              ψ0s[t], V, problem.times[t]),
            length(ψ0s))
        finals = ntuple(t -> trajs[t].states[end], length(trajs))
        μhat = apply_measurement(problem.measurement, finals)
        r = μhat .- μobs
        length(r) == length(μhat) || throw(ArgumentError(
            "measurement size mismatch for experiment $i"))
        J += sum(abs2, r) / σ²

        n_s = length(apply_measurement(problem.measurement, (finals[1],)))
        length(r) == n_s * length(finals) || throw(ArgumentError(
            "measurement layout mismatch for experiment $i"))

        for t in eachindex(finals)
            traj = trajs[t]
            rt = r[(t-1)*n_s+1:t*n_s]
            λ = measurement_adjoint(problem.measurement, finals[t], rt, σ²)

            for j in length(traj.states)-1:-1:1
                ψj = traj.states[j]
                # b = Kᴴ λ; then the θ-gradient is 2 Re(Φᵀ c).
                b = ifft(conj.(traj.Kfac) .* fft(λ))
                c = conj.(b) .* (-im .* traj.dt .* (traj.Vfac .* ψj))
                g .+= 2 .* real.(transpose(Φ) * vec(c))
                # λ_j = Uᴴ λ_{j+1} = Vᴴ b.
                λ = conj.(traj.Vfac) .* b
            end
        end
    end

    J /= m
    g ./= m
    J += ε * regularizer_value(regularizer, θ)
    g .+= ε .* regularizer_gradient(regularizer, θ)
    return J, g
end
