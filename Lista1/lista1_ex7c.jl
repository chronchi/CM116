#vamos escrever uma função que recebe os parâmetros dos modelos,
#e os dados de teste
#este programa requer o pacote MLBase

using MLBase

function onevsall(parametro, testex, testey)
  m,n = size(testex)
  p = size(parametro,2)
  previsto = []
  for i = 1:m
    a = []
    for k = 1:p
      a = [a;sigm(dot(parametro[:,k],testex[i,:]))]
    end
    b = find(a .== maximum(a))
    previsto = [previsto; b]
  end
  return convert(Array{Int64}, previsto)
end

#A taxa de acerto para os modelos criados pelo método do gradiente é dado abaixo

previsto = onevsall(parameters, xtest, ytest)
corr = correctrate(ytest,previsto)
println("A taxa de acerto para os modelos estimados através do método do gradiente
é de $(100*corr)%")

#A taxa de acerto para os modelos criados pelo método do gradiente acelerado
#de Nesterov é dado abaixo

previsto = onevsall(parameters, xtest, ytest)
corr = correctrate(ytest,previsto)
println("A taxa de acerto para os modelos estimados através do método do gradiente
acelerado de Nesterov é de $(100*corr)%")
