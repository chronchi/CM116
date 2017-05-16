include("lista1_ex2a.jl")
using DataFrames

#Função que calcula o cos(α), onde α é o ângulo entre os vetores u,v.
function scos(u,v)
  b = norm(u) * norm(v)
  a = dot(u,v)
  return a/b
end

#Função do modelo da média aritmética ponderada para o sistema de recomendação
#do exercício 2-b.
function medap(u::Vector, tabela)
  #Inicializando uma matriz para conter as notas dos usuários diferentes de u.
  X = zeros(6,5)
  #Procurando onde se encontra o usuário na tabela de notas
  for j = 1:6
    if tabela[:,j] == u
      if j == 1
        X = tabela[:,2:6]
      elseif j == 6
        X = tabela[:,1:5]
      else
        X = [tabela[:,1:(j-1)] tabela[:,(j+1):6]]
      end
    end
  end
  #Retorna as posições no vetor cuja as notas não foram dadas ainda.
  a = find(u .== 0.0)

  #vamos determinar o ranking agora para cada
  #coordenada do vetor u que possui valor 0.0
  #pois são os filmes não avaliados pela pessoa

  notas = zeros(length(a))
  count = 1
  for i in a
    c = 0
    d = 0
    for k = 1:5
      list_u = []
      list_v = []
      #Procura pelas coordenadas diferentes de 0 para u e os outros usuários
      for j = 1:6
        if u[j] != 0.0 && X[j,k] != 0.0
          list_u = [list_u; u[j]]
          list_v = [list_v; X[j,k]]
        end
      end
      if X[i,k] != 0.0
        c = c + scos(list_u,list_v)
      end
      d = d + scos(list_u,list_v) * X[i,k]
    end
    e = d/c
    notas[count] = e
    count += 1
  end
  return notas
end

#Vamos criar uma tabela com o nome dos filmes e pessoas
#com as respectivas notas, previstas e já dadas utilizando o
#pacote DataFrames.

filmes = ["Sem Retorno"; "Para Sempre Alice";
"Curitiba Zero Grau"; "Como eu era antes de você";
"Forrest Gump"; "O Silêncio dos Inocentes"]
avaliadores = ["Natalha"; "Karla"; "Evelin";
"Aline"; "Geovani"; "Abel"]
df = DataFrame();
df[:,1] = filmes
counter = 0
#vamos determinar as notas para os usuários ainda não estimados
#no exercício anterior.
for i in [1,4,5,6]
  counter += 1
  notas = medap(tabela[:,i],tabela)
  a = find(tabela[:,i] .== 0.0)
  b = zeros(6)
  #procurando na "tabela" onde tem valores iguais a 0 para salvar em um vetor e
  #depois adicionar à tabela df
  for k = 1:6
    if k in a
      b[k] = notas[find(a .== k)[1]]
    else
      b[k] = tabela[k,i]
    end
  end
  #adicionando as notas obtidas na tabela df.
  if counter == 1
    df[:,i+1] = b
  else
    df[:,counter+1] = b
  end
end
#Juntando as notas obtidas na letra a e b

df[:,6] = Notas_kar
df[:,7] = Notas_eve
nomes = ["Filmes"; avaliadores]
names!(df, [Symbol(nomes[i]) for i in [1,2,5,6,7,3,4]])
