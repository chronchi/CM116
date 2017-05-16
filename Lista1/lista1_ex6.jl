function metgradnest(grad::Function,x::Vector, L::Float64; tol::Float64 = 1e-4,
  max_iter = 1000)

  λ = 0
  iter = 0
  y = x
  gr = grad(x)
  while norm(grad(x)) > 0 && iter < max_iter
    y_new = x - grad(x)/L
    λ_new = (1+sqrt(1+4*λ^2))/2
    γ = (1-λ)/λ_new
    λ = λ_new
    x = (1-γ)*y_new + γ*y
    y = y_new
    iter += 1
  end
  return y, iter
end
