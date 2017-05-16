# grad é o gradiente da função a ser minimizada
# x é um ponto inicial qualquer
# L é uma cota superior para a constante de Lipschitz do gradiente

function metgrad(grad,x, L; tol = 1e-4,max_iter = 10000)
  iter = 0
  gr = grad(x)
  while norm(gr) > tol && iter < max_iter
    x = x - grad(x)/L
    gr = grad(x)
    iter += 1
  end
  return x, iter
end
