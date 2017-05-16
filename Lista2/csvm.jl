#Função c-SVM como visto na aula

function csvm(xtrain::Matrix,ytrain::Vector,c::Float64)

  #Adiciona a coluna do intercepto
  X = [ones(size(xtrain,1)) xtrain]

  m,n = size(X)

  #Vamos escrever a matriz e o vetor da função quadrática f(x) = 0.5x'*H*x + f'*x

  #Matriz H da função quadrática acima
  H = zeros(m+n,m+n)
  H[2:n,2:n] = eye(n-1)

  #vetor f da função quadrática acima
  f = zeros(m+n)

  #O "c" é o valor dado para o modelo c-SVM
  f[(n+1):end] = c .* ones(m)

  #Vamos escrever as restrições para o c-SVM

  #Para a restrição Ax <= b temos
  A = -[(ytrain.*X) eye(m)]
  b = -ones(m)

  #Para a limitação inferior do parâmetro ξ
  lb = [[-1e16 for i =1:n]; zeros(m)]

  #quadprog é a função que retorna o valor ótimo para este problema de otimização
  #com restrições
  return θ = quadprog(H,f,A=A,b=b,lb=lb)
end
