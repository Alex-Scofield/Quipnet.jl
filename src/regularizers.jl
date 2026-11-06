abstract type AbstractRegularizer end

"""Tikhonov penalty `R(θ) = ‖θ‖²`."""
struct L2Regularizer <: AbstractRegularizer end

regularizer_value(::L2Regularizer, θ) = sum(abs2, θ)
regularizer_gradient(::L2Regularizer, θ) = 2 .* θ
