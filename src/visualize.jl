function animate_free_particle_periodic(ψ₀, x, tmax; filename="free.gif", V=nothing, nsteps=200)
    n = 180
    prop = V === nothing ? (t -> free_schrod_periodic(ψ₀, x, t)) :
                           (t -> schrod_periodic(ψ₀, V, x, t; nsteps=nsteps))
    dens = [abs2.(prop(t)) for t ∈ range(0, tmax, n)]
    ymax = maximum(maximum, dens)
    anim = @animate for d in dens
        plot(x, d, ylims=(0, ymax))
    end
    gif(anim, filename, fps = 30)
end

function animate_free_particle_periodic(ψ₀, x, y, tmax; filename="free.gif", V=nothing, nsteps=200)
    n = 120
    prop = V === nothing ? (t -> free_schrod_periodic(ψ₀, (x, y), t)) :
                           (t -> schrod_periodic(ψ₀, V, (x, y), t; nsteps=nsteps))
    sols = [prop(t) for t ∈ range(0, tmax, n)]
    cmax = maximum(maximum(abs2, s) for s in sols)
    anim = @animate for s in sols
        dens = permutedims(abs2.(s))
        heatmap(x, y, dens, aspect_ratio=:equal, clims=(0, cmax))
    end
    gif(anim, filename, fps = 30)
end
