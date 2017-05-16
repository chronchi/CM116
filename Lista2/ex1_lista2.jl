#Para este exercício utilizaremos os pacotes DataFrames, Gadfly. Caso não os tenha,
#basta digitar no REPL: Pkg.add("DataFrames") e Pkg.add("Gadfly")

#pacotes para tabelar os dados e para visualiza-los também
using DataFrames, Gadfly

#Formatando os dados para deixa-los como na lista
dados = DataFrame(ANO = collect(2003:2016), IPCA = [9.30;7.60; 5.69;3.14;4.46;
5.90;4.31;5.91;6.50;5.84;5.91;6.41;10.67;6.29], SELIC = [17.05;17.51;18.23;
13.25;11.25;13.75;9.05;10.75;11.00;7.25;10.00;11.70;14.25;13.75])

#Alguns plots sobre os dados

#Observe que com o plot a seguir fica fácil de ver que os maiores valores da taxa
#selic e ipca juntamente ocorreram nos primeiros anos, em 2003,2004.
plot(dados,x=:IPCA,y=:SELIC, color=:ANO)

#Nos 4 primeiros anos houve uma queda acentuada do IPCA, após isso houve uma estabilidade,
#com exceção do ano de 2015, que o IPCA foi o maior registrado de 2003 a 2016
plot(dados,x=:ANO,y=:IPCA)

#Após 2005, os valores da Selic ficaram variando entre 5 e 15%.
plot(dados,x=:ANO,y=:SELIC)

#Após essa breve análise dos dados, vamos montar a matriz de dados para
#podermos aplicar a rede neural

#Inicializando a matriz de dados e o vetor de classificação

xtrain = zeros(0,6)
ytrain = zeros(0)

#adicionando os valores na matriz, temos que

u,v = size(dados) #tamanho da matriz de dados, contando a coluna do Ano

n = 11 #número de linhas da matriz de dados

m = 3  #número de variaveis em relação ao IPCA

r = 3  #número de variaveis em relação à taxa Selic

for i in collect(u:-1:(u-n+1))
   xtrain = vcat(xtrain, [dados[:IPCA][(i-1):-1:(i-m)]' dados[:SELIC][(i-1):-1:(i-m)]'])
   ytrain = vcat(ytrain,dados[:IPCA][i])
end

#Note que do enunciado do exercício, não adicionamos uma coluna com 1's para o intercepto
#do coeficiente que vamos calcular

#Antes de utilizar redes neurais, vamos encontrar o valor utilizando minimos quadrados

θ = (xtrain'*xtrain)\(xtrain'*ytrain)

#Prevendo o valor do IPCA para 2017, temos que

d = 15
xtest = [dados[:IPCA][(d-1):-1:(d-m)];dados[:SELIC][(d-1):-1:(d-m)]]

valor_previsto = θ'*xtest

#O valor previsto foi de 7.1912%, um valor distante da projeção feita
#pelo mercado financeiro. Vamos determinar agora o coeficiente utilizando
#uma rede neural com uma única camada escondaida contendo três neurônios
#e vamos ver se obtemos um parâmetro que se ajusta melhor.

g(x) = 1/(1+exp(-x))
#Seja s a função vista na sala de aula
#W é a matriz com os pesos da primeira camada para a segunda
#z,b,c são os valores vistos na aula, enquanto x é uma entrada para os valores dos dados
sfun(W,z,b,c,x) = dot(g.(W*x + b),z) + c

#A função abaixo calcula o erro de uma rede neural para um dado apenas
#para todos os dados basta somar os erros
E(W,z,b,c,x,y) = 0.5*norm(sfun(W,z,b,c,x) - y)^2

#vamos escrever uma função que calcula o gradiente da função Erro acima. Note que
#adicionamos os valores x e y como os dados para a função erro, mas apenas para facilitar
#a implementação do código.

function backprop(x, y, W, b, z, c)
  m = length(x)

  Wgrad = zeros(size(W))
  bgrad = zeros(length(b))
  zgrad = zeros(length(z))

  sapp = sfun(W,z,b,c,x)

  for i = 1:size(W,1)
    sig = g(dot(W[i,:],x)+b[i])
    for j = 1:size(W,2)
      Wgrad[i,j] = (sapp - y)*z[i]*sig*(1-sig)*x[j]
    end
    bgrad[i] = (sapp - y)*z[i]*sig
    zgrad[i] = (sapp - y)*sig
  end

  cgrad = sapp-y

  return Wgrad,bgrad,zgrad,cgrad
end

#agora vamos implementar a rede neural com 3 neurônios na camada escondida

#xtrain e ytrain são os dados, nl é o número de camadas da rede neural e o tamanho
#de cada camada
function nn(xtrain::Matrix, ytrain::Vector, nl::Vector; step_size::Float64 = 0.3,
epoch::Int64 = 10, returnmed=false)
  m,n = size(xtrain)

  #Inicialização dos parâmetros
  W = rand(nl[2],nl[1])
  b = rand(nl[2])
  z = rand(nl[2])
  c = rand(nl[3])[1]

  Wend = zeros(size(W))
  bend = zeros(size(b))
  zend = zeros(size(z))
  cend = 0.0

 iter = 0
 evaluated = zeros(0,2)

 #vamos calcular agora a atualização do parâmetro calculando o gradiente
 #com a função backprop. Vamos iterar várias vezes sobre os dados, tomando um
 #dado por vez para o cálculo
 random = sample(1:m,m,replace=false)
 for i in 1:epoch
   for numb in random
     Wgrad,bgrad,zgrad,cgrad = backprop(xtrain[numb,:], ytrain[numb], W, b, z, c)
     W -= step_size*Wgrad
     b -= step_size*bgrad
     z -= step_size*zgrad
     c -= step_size*cgrad
     if iter%10 == 0
       eval = 0
       for train in 1:m
         eval += E(W,z,b,c,xtrain[train,:], ytrain[train,:])
       end
       evaluated = vcat(evaluated, [eval iter])
     end
     iter += 1
     Wend += W
     bend += b
     zend += z
     cend += c
   end
 end
 if returnmed == true
   return Wend./iter,bend./iter,zend./iter,cend./iter
 else
   return W,b,z,c
 end
end

#Com a rede neural feita, podemos usa-la para encontrar os parâmetros W,b,z,c
#vamos usar o método multistart
par = []

multistart = 500

for i = 1:multistart
  par = [par;nn(xtrain,ytrain,[6;3;1],step_size = 0.003, epoch = 10,returnmed=true)];
end

#vamos ver qual desses valores é o melhor valor

evaluated = []
for train in 1:multistart
  eval = 0
  for i in 1:m
    eval += E(par[train][1],par[train][2],par[train][3],par[train][4],xtrain[i,:], ytrain[i,:])
  end
  evaluated = [evaluated; eval]
end

b = find(evaluated.==minimum(evaluated))[1]

println("O menor valor para a função objetivo é dado pelos parâmetros em $b")


#Logo, o resultado encontrado pelo modelo é
resultado = sfun(par[b][1],par[b][2],par[b][3],par[b][4],xtest)

info("O modelo previu que a projeção do IPCA para 2017 é $resultado")

#Observe que o valor que o modelo gera pode variar a cada vez que se executa a função
#da rede neural para obter os parâmetros. Isto acontece devido a inicialização aleatória
#dos dados toda vez que a função é executada.
