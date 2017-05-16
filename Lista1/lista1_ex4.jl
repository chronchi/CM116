include("lista1_ex3.jl")

using DataFrames

#Lendo os dados da Taxa Selic

data = readtable("selic.csv")

data_train = data[212:-1:1,:]
data_test = data[266:-1:213,:]

#vamos agora montar o sistema AR(6)


A_train = zeros(206,7);
y_train = convert(Array{Float64},data_train[1:206,2])
m,n = size(A_train)

for i in 1:m
  for j = 1:n
    if j == 1
      A_train[i,j] = 1.0
    else
      A_train[i,j] = data_train[i+j-1,2]
    end
  end
end

#vamos determinar o parametro θ utilizando o
#método dos gradientes conjugados com a função quadratica e convexa
#f(θ) = (1/2)*Θ'(2*A'A)θ + (-2*A'y)'θ + y'y
#note que gradiente de f é 2A'(Aθ - y)
g(alpha) = 2*A_train'*(A_train*alpha  - y_train)
x_train = rand(7)
println("hi")
θ = gradconj(g, 2*A_train'*A_train, x_train,tol=1e-6)
println("bye")

y_test = data_test[1:48,2]
A_test = zeros(48,7)

m,n = size(A_test)
for i in 1:m
  for j = 1:n
    if j == 1
      A_test[i,j] = 1.0
    else
      A_test[i,j] = data_test[i+j-1,2]
    end
  end
end

eq = 0.0;
for i = 1:m
  eq += ((dot(A_test[i,:], θ) - y_test[i])^2)/y_test[i]
end

eqm = eq/m
