using GLMakie
using DelimitedFiles
using DataInterpolations

straub_raw = readdlm("ar_xsection_straub.txt", '\t')
straub = [(x == "" ? 0.0 : float(x)) for x in straub_raw[2:end, :]]
straub_header = straub_raw[1, :]
units = map(x -> split(split(x, '(')[2], ')')[1], straub_header[2:end])
scales = map(x -> parse(Float64, replace(split(x)[1], "10^" => "1e")), units)
unit_conversions = map(x -> split(x)[2] == "cm^2" ? 1e-4 : 1.0, units)


ionization_energy = 15.76
straub = [[ionization_energy 0 0 0]; straub]
energy_straub = straub[:, 1]
sigma_straub = map(i -> straub[:, i+1] * scales[i] * unit_conversions[i], 1:3)

f = Figure()
ax = Axis(f[1, 1], xscale=log10, yscale=log10, xlabel="Electron energy [eV]", ylabel="Cross section [Å²]")
ylims!(ax, 2e-4, 5)
lines!(ax, energy_straub, sigma_straub[1] * 1e20)
lines!(ax, energy_straub, sigma_straub[2] * 1e20)
lines!(ax, energy_straub, sigma_straub[3] * 1e20)
f

energies = 0:255
rate_coeffs = [zeros(length(energies)) for _ in 1:3]
q_e = 1.60217663e-19
m_e = 9.1093837e-31
num_samples = 50_000

table = zeros(length(energies), 4)
table[:, 1] = energies

function compute_rate_coeff(sample_E, sample_sigma, E; num_samples=100_000)
    itp = LinearInterpolation(sample_sigma * 1e20, sample_E, extrapolate=true)
    samples = randn(3, num_samples)
    kiz = zeros(length(E))
    for (i, e) in enumerate(E)
        Te = 2 / 3 * e
        thermal_speed = sqrt(q_e * Te / m_e)

        kiz[i] = 0.0
        for j in 1:num_samples
            speed_squared = (samples[1, j]^2 + samples[2, j]^2 + samples[3, j]^2) * thermal_speed^2
            speed = sqrt(speed_squared)
            energy_eV = 0.5 * m_e * speed_squared / q_e
            sigma = max(0.0, itp(energy_eV))
            if (e == 20.0 && j < 10)
                @show energy_eV, sigma
            end
            kiz[i] += speed * sigma
        end
        kiz[i] /= num_samples
    end
    kiz /= 1e20
    return kiz
end

@time table[:, 2] .= compute_rate_coeff(energy_straub, sigma_straub[1], energies; num_samples)
@time table[:, 3] .= compute_rate_coeff(energy_straub, sigma_straub[2], energies; num_samples)
@time table[:, 4] .= compute_rate_coeff(energy_straub, sigma_straub[3], energies; num_samples)

f2 = Figure()
ax2 = Axis(f2[1, 1], yscale=log10, xlabel="Mean electron energy [eV]", ylabel="Ionization rate coefficient [Å²m/s]")
ylims!(ax2, 0.005, 100)
lines!(ax2, energies, table[:, 2] * 1e20)
lines!(ax2, energies, table[:, 3] * 1e20)
lines!(ax2, energies, table[:, 4] * 1e20)
f2

ionization_energies = [15.75962, 27.62967, 40.74]
open("ionization_Ar_Ar+.dat", "w") do f
    write(f, "Ionization energy (eV): $(ionization_energies[1])\n")
    write(f, "Energy (eV)\tRate coefficient (m/s)\n")
    writedlm(f, table[:, 1:2])
end

open("ionization_Ar_Ar2+.dat", "w") do f
    write(f, "Ionization energy (eV): $(ionization_energies[1] + ionization_energies[2])\n")
    write(f, "Energy (eV)\tRate coefficient (m/s)\n")
    writedlm(f, table[:, [1, 3]])
end

open("ionization_Ar_Ar3+.dat", "w") do f
    write(f, "Ionization energy (eV): $(sum(ionization_energies))\n")
    write(f, "Energy (eV)\tRate coefficient (m/s)\n")
    writedlm(f, table[:, [1, 4]])
end
