#método dos gradiente conjugados
#Para uma função da forma f(x) = c + b'x + x'Ax
#é necessário apenas a matriz A e o vetor b
#como entradas. Mas vamos utilizar como entrada o
#gradiente da função f, a dimensão do dominio de f e A.

function gradconj(grad::Function, A, x::Vector; tol = 1e-4,max_iter = 1000)
  k = 0
  β = 0.0
  t = 0.0
  d = -grad(x)
  while norm(grad(x)) > tol && k < max_iter
    grad_d_dot = -dot(grad(x),d)
    conj_d = dot(d,A*d)
    t = grad_d_dot/conj_d
    x = x + t*d
    conj_d_grad = dot(A*grad(x),d)
    β = conj_d_grad/conj_d
    d = -grad(x) + β*d
    k = k + 1
    if k%100 == 0
      println(norm(grad(x)))
    end
  end
  return x
end
