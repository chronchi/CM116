#Função do método de k-means, onde k é o número de classes. x é
#a matriz com os dados de entrada para a clusterização

#Função argmin para kmean
function argmin(x,y)
    m,n = size(y)
    vector = zeros(0)
    for i = 1:m
      vector = [vector;norm(x-y[i,:])]
    end

    minvalue = find(vector.== minimum(vector))

    #caso haja duas posições de mesmo valor em vector
    #selecionamos uma delas aleatoriamente
    if length(minvalue) > 1
        minvalue = sample(minvalue,1)
    end

    if length(minvalue) == 1
        minvalue = minvalue[1]
    end

    return minvalue
end

function summatrixrow(x::Matrix)
  m,n = size(x)
  vector = zeros(n,1)
  for j = 1:m
    vector += x[j,:]
  end
  return vector
end

function kmean(x,k; max_iter=1000)

    m,n = size(x)

    #Passo 0: retirar k pontos aleatoriamento dos dados de entrada
    random_mean = sample(1:m,k,replace=false)
    x_mean =  x[random_mean,:]

    #Passo 1: Determinar de quais médias cada valor de entrada
    #está mais próximo


    t = 0

    for p = 1:max_iter

        list = zeros(Int64,0)

        for i = 1:m

            #vamos adicionar em list o vetor e à classe que ele pertence
            list = [list; argmin(x[i,:],x_mean)]

            #Passo 2: se t>1 e S(t) = S(t+1), pare.

        end

        #Passo 3: Atualização das médias

        for means = 1:k
          whereis = find(list .== means)
          if length(whereis) == 0
            continue
          end
          x_mean[means,:] = (1/length(whereis)) * summatrixrow(x[whereis,:])
        end
    end

  return x_mean
end

#Para comparar o método implementado com um já feito na linguagem Julia, vamos
#utilizar o pacote Clustering, caso não o tenha, basta digitar
#Pkg.add("Clustering") no REPL.

#Vamos utilizar o banco de dados iris para efeitos de comparação

using Clustering
using RDatasets
using DataFrames

iris = dataset("datasets", "iris")

x = convert(Array{Float64}, iris[:,1:4])

#para o método implementado acima
result1 = kmean(x,3,max_iter=200);

#para o método já implementado no julia disponivel com o pacote Clustering
result2 = kmeans(x',3);

#Note que os valores são bem proximos

result1'
M = result2.centers
