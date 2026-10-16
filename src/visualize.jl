function animate_free_particle_periodic(ψ₀, x, tmax; filename="free.gif", V=nothing, nsteps=200)
    n = 180
    prop = V === nothing ? (t -> free_schrod_periodic(ψ₀, x, t)) :
                           (t -> schrod_periodic(ψ₀, V, x, t; nsteps=nsteps))
    anim = @animate for t ∈ range(0, tmax, n)
        plot(x, abs2.(prop(t)))
    end
    gif(anim, filename, fps = 30)
end

function animate_free_particle_periodic(ψ₀, x, y, tmax; filename="free.gif", V=nothing, nsteps=200)
    n = 120
    prop = V === nothing ? (t -> free_schrod_periodic(ψ₀, (x, y), t)) :
                           (t -> schrod_periodic(ψ₀, V, (x, y), t; nsteps=nsteps))
    anim = @animate for t ∈ range(0, tmax, n)
        dens = permutedims(abs2.(prop(t)))
        heatmap(x, y, dens, aspect_ratio=:equal, clims=(0, maximum(dens)))
    end
    gif(anim, filename, fps = 30)
end
