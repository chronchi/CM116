#Para este exercício utilizaremos os pacotes DataFrames, Gadfly. Caso não os tenha,
#basta digitar no REPL: Pkg.add("DataFrames") e Pkg.add("Gadfly")

#pacotes para tabelar os dados e para visualiza-los também
using DataFrames, Gadfly

#Lendo os dados do arquivo ex2_lista2dataset.csv
ex2data = readtable("ex2_lista2dataset.csv")


#Vamos selecionar as amostras

#Primeiro vamos criar um vetor de inteiros com 550 entradas com valores entre
#1 e 699

#tamanho dos dados
m,n = size(ex2data)
#tamanho do conjunto de treinamento
p = 550

random_vector = sample(1:m,p,replace=false)

#Portanto podemos dividir os nossos dados em duas partes, na parte de treinamento
#e na parte de teste

#os dados de treinamento são
Train = ex2data[random_vector,:]

#criando a matriz para utilizar os dados de treinamento nos modelos

xtrain = convert(Array{Float64}, Train[:,2:10])
ytrain = convert(Array{Float64}, Train[:,11])

#transformando as classes 2 e 4 em 1 e -1

for i = 1:length(ytrain)
  if ytrain[i] == 2.0
    ytrain[i] = -1.0
  else
    ytrain[i] = 1.0
  end
end

#Vamos normalizar os dados também

for i = 1:size(xtrain,2)
  a,b = mean(xtrain[:,i]), sqrt(var(xtrain[:,i]))
  xtrain[:,i] = (xtrain[:,i] .- a)./b
end

#e os dados de teste são
crandom_vector = setdiff(collect(1:m),random_vector)
Testdata = ex2data[crandom_vector,:]

#Como acima, formatamos os dados de teste também para

xtest = convert(Array{Float64}, Testdata[:,2:10])
ytest = convert(Array{Float64}, Testdata[:,11])

for i = 1:length(ytest)
  if Testdata[i,11] == 2.0
    ytest[i] = -1.0
  else
    ytest[i] = 1.0
  end
end

for i = 1:size(xtest,2)
  a,b = mean(xtest[:,i]), sqrt(var(xtest[:,i]))
  xtest[:,i] = (xtest[:,i] .- a)./b
end

#Com isso, finalizamos a letra a do exercício 2.

#Para a letra b do exercício 2, vamos calcular os modelos,
#começando com o c-svm

#abrindo as funções que serão utilizadas
include("minquad.jl")
include("csvm.jl")

#Como os dados já estão formatados, vamos calcular os parâmetros
#utilizando o C-SVM com C igual a 0.001, 1.0, 10.0, 100.0

C = [0.01; 1.0; 10.0; 100.0]

#inicializando a matriz dos parâmetros para salvar o parâmetro
#para cada C.

parameterscsvm = zeros(0,(n-1)) #(n-1) é o número de colunas da matriz
                            #de dados acrescido da coluna com os 1's

#Aplicando agora o csvm para cada c e salvando os parâmetros

for i in C
  parameterscsvm = vcat(parameterscsvm, csvm(xtrain,ytrain,i)[1:(n-1)]')
end

#Vamos agora calcular os valores para o modelo de regressão logistica
#já implementado na lista 1. Como temos apenas duas classes, precisamos
#calcular apenas um parâmetro

#Note que para o modelo de regressão logistica, devemos usar
#0 e 1 como valores para as classes, Portanto

for i = 1:length(Train[:,11])
  if Train[i,11] == 2
    ytrain[i] = 0.0
  else
    ytrain[i] = 1.0
  end
end

#Montando o modelo de regressão logística para essas duas classes e
#utilizando o método do gradiente acelerado de Nesterov para encontrar o melhor
#parâmetro

include("lista1_ex6.jl")

#adicionando a coluna de 1's na matriz xtrain

xtrain = [ones(size(xtrain,1)) xtrain]

L = 2.0

#função sigm(x) = 1/(1+exp(-x))

sigm(x) = 1/(1+exp(-x))

grad(x) = xtrain'*(sigm.(xtrain*x)-ytrain)
theta, iter = metgradnest(grad,zeros(10),L,max_iter=50000)

#Com isso, calculamos os 4 parâmetros para o CSVM e um
#parâmetro para o modelo de regressão logística, concluindo a letra b.

#Agora para a letra c do exercício 2, vamos salvar a quantidade de acertos
#de cada modelo no vetor resultados

resultados = zeros(0)

#juntando os parâmetros

parametros = [parameterscsvm; theta']

for j = collect(1:size(parametros,1))
  count = 0
  if j == size(parametros,1)
    for i = 1:length(Testdata[:,11])
      if Testdata[i,11] == 2
        ytest[i] = 0.0
      else
        ytest[i] = 1.0
      end
    end
    result = zeros(length(ytest))
    for i = 1:length(ytest)
      result[i] = convert(Int64,round(sigm(dot(parametros[j,:],[1;xtest[i,:]]))))
    end
    result = sum(result.==ytest)
    resultados = [resultados;result]
  else
    result = zeros(length(ytest))
    for i = 1:length(ytest)
      if dot(parametros[j,:],[1;xtest[i,:]]) >= 0.0
        result[i] = 1.0
      else
        result[i] = -1.0
      end
    end
    for i = 1:length(ytest)
      if Testdata[i,11] == 2.0
        ytest[i] = -1.0
      else
        ytest[i] = 1.0
      end
    end
    result = sum(result.==ytest)
    resultados = [resultados;result]
  end
end

resultados = convert(Array{Int64}, resultados)

#Portanto, temos que a acurácia dos modelos é dada por

for j = 1:5
  info("O modelo $j apresentou uma acurácia de $(
  resultados[j]./length(ytest)))")
end
