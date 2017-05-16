include("lista1_ex5.jl")

using DataFrames
using RDatasets
using StatsBase

#Carregando os dados
iris = dataset("datasets", "iris");

#Convertendo os dados para poder aplica-los nos métodos

xdata = convert(Array{Float64}, iris[:,1:4]);
ydata = zeros(150);

for i = 1:150
  if iris[i,5] == "setosa"
    ydata[i] = 1.0
  elseif iris[i,5] == "versicolor"
    ydata[i] = 2.0
  else
    ydata[i] = 3.0
  end
end

#Com os dados convertidos vamos criar o conjunto de treinamento e de testes

xtest = zeros(30,5);
ytest = zeros(30);

#Gera um vetor de indices aleatórios entre 1 e 150
random_vector = [sample(1:50,10,replace=false); sample(51:100,10,replace=false);sample(101:150,10,replace=false)];

#Dados para teste
for i = 1:30
  xtest[i,:] = [1 xdata[random_vector[i],:]']
  ytest[i] = ydata[random_vector[i]]
end

ytest = convert(Array{Int64}, ytest)
#Dados para o treinamento
xtrain = [ones(120) xdata[setdiff(1:end,random_vector),:]];
ytrain = convert(Array{Int64},ydata[setdiff(1:end,random_vector)]);

#Vamos normalizar os dados

for j = 2:5
  for i = 1:120
    xtrain[i,j] = (xtrain[i,j] - mean(xtrain[:,j]))/var(xtrain[:,j])
  end
  for i = 1:30
    xtest[i,j] = (xtest[i,j] - mean(xtest[:,j]))/var(xtest[:,j])
  end
end
#Inicializando a matriz dos parâmetros
parameters = zeros(5,3);

#Função sigmoide
sigm(z) = 1/(1+exp(-z))
#Ponto inicial para os métodos do gradiente e gradiente aceleradot
thetarand =  [0.475541; 0.626087; 0.768106; 0.555628; 0.504413]

#Cota superior da constante de lipschitz
L = norm(xtrain'*xtrain)
iterlist = []

for i = 1:3
  ycon = zeros(120);
  for j = 1:120
    if ytrain[j] == i
      ycon[j] = 1
    end
  end
  grad(theta) = xtrain'*(sigm.(xtrain*theta)-ycon)
  thetai,iter = metgrad(grad,thetarand,L,max_iter = 10000)
  iterlist = [iterlist; iter]
  parameters[:,i] = thetai
end

#Agora que temos os parâmetros, vamos determinar o erro no conjunto de testes

for k = 1:3
  count = 0
  ycon = zeros(30);
  for j = 1:30
    if ytest[j] == k
      ycon[j] = 1
    end
  end
  for i = 1:30
    a = round(sigm(dot(parameters[:,k],xtest[i,:])))
    if a == ycon[i]
      count += 1
    else
    end
  end
  println("O modelo para o $k parâmetro apresenta uma taxa de acerto
  de $(100*count/30)%")
end
