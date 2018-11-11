using Distributed

function problem(x1::Float64, x2::Float64)
    x1^2 + x2^2
end

const Maximization = >
const Minimization = <

mutable struct Target
    problem::Function
    best_than::Function
    xdefzone::Array{Tuple{Float64,Float64},1}
    

    global_best::Vector{Float64}     # 截止目前，所有粒子获得的最优解
    global_value::Float64

    Target(p::Function, compare::Function, n::Int64, xdef::Array{Tuple{Float64,Float64},1})  = 
            new(p, compare, xdef, Vector{Float64}(undef, n), ifelse(compare == Maximization, -Inf, Inf))
end

mutable struct SwarmParams
    population::Int64      # 群体数量
    iteration::Int64       # 迭代次数
    c1::Float64             # 学习因子c1
    c2::Float64             # 学习因子c2
    w::Float64              # 惯性因子

    SwarmParams(p::Int64, i::Int64, c1::Float64, c2::Float64, w::Float64) = new(p, i, c1, c2, w)
end

mutable struct Particle
    position::Vector{Float64}       # 粒子的当前位置
    velocity::Vector{Float64}       # 粒子的当前速度

    local_best_position::Vector{Float64}     # 截止目前，该粒子获得的最优解
    local_best_value::Float64

    Particle(n::Int64) = new(zeros(Float64,n), zeros(Float64,n), zeros(Float64,n), 0.0)
end


function update_particle(p::Particle, sc::SwarmParams, t::Target)
    p.velocity = sc.w .* p.velocity + sc.c1 * rand() .* (p.local_best_position .- p.position)+ sc.c2 * rand() .* (t.global_best .- p.position)

    @sync @distributed for i in 1:length(p.velocity)
        if p.velocity[i] > target.xdefzone[i][2] || p.velocity[i] < target.xdefzone[i][1]
            p.velocity[i] = ifelse(rand() > 0.5, -1.0, 1.0) * rand()
        end
    end

    p.position = p.position .+ p.velocity

    @sync @distributed for i in 1:length(p.position)
        if p.position[i] > target.xdefzone[i][2] || p.position[i] < target.xdefzone[i][1]
            p.position[i] = rand() .* (target.xdefzone[i][2]-target.xdefzone[i][1]) .+ target.xdefzone[i][1]
        end
    end
end

function optimize(target::Target, sc::SwarmParams)
    @show target.xdefzone

    dimension = length(target.global_best)

    swarm = Vector{Particle}()
    for i in 1:sc.population
        push!(swarm,Particle(dimension))
    end

    @sync @distributed for p in swarm
        p.velocity = rand(dimension)
        p.position = rand(dimension)

        for i in 1:length(p.position)
            p.position[i] = p.position[i] .* (target.xdefzone[i][2]-target.xdefzone[i][1]) .+ target.xdefzone[i][1]
        end

        p.local_best_position = p.position

        p.local_best_value = target.problem(p.position...)
    end

    println(">>> ---- starting ----")

    for i in 1:sc.iteration
        @show i
        @sync @distributed for p in swarm
            value = target.problem(p.position...)
            @show value

            if target.best_than(value, p.local_best_value)
                p.local_best_value = value
                p.local_best_position = p.position
            end
        end

        @sync @distributed for p in swarm
            if target.best_than(p.local_best_value, target.global_value)
                target.global_value = p.local_best_value
                target.global_best = p.position
            end
        end

        @sync @distributed for p in swarm
            update_particle(p, sc, target)
        end
    end
end

target = Target(problem, <, 2, [(-10.0, 10.0), (-10.0, 10.0)])
sc = SwarmParams(20, 500, 2.0, 2.0, 1.0)

optimize(target, sc)

println(target.global_value)
println(target.global_best)