function choleskyfact(A::Matrix)
  #Cria-se uma cópia de A para fazer modificações
  R = copy(A)
  m = size(R,1)
  for i = 1:m
    #Cria-se a matriz C para criar a submatriz de R segundo a fatoração de cholesky
    C = zeros(m-(i-1),m-(i-1))
    C[1,1] = sqrt(R[i,i])
    C[2:end,1] = R[(i+1):end,i]./C[1,1]
    C[2:end,2:end] = R[(i+1):end,(i+1):end] - C[2:end,1]*C[2:end,1]'
    R[(i:m),(i:m)] = C
  end
  return R
end

function quad_min(x,y)
  m = size(x,1)
  # Matriz dos dados
  X = [ones(m) x]
  b = X' * y
  # Sistema fica da forma X'Xa=b
  A = choleskyfact(X'*X)
  # AA'a=b
  # Vamos resolver primeiro Ac=b
  c = zeros(length(b))
  for i = 1:length(b)
    c[i] = ((b[i] - A[i,:]'*c)/A[i,i])[1]
  end
  # Encontrando o valor de a no sistema A'a = c
  a = zeros(length(c))
  for i in collect(1:length(a))[end:-1:1]
    a[i] = ((c[i] - A'[i,:]'*a)/A'[i,i])[1]
  end
  return a
end
