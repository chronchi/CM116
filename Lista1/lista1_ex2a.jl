# queremos prever a nota dos filmes com base nas notas ja atribuidas por
# quem assistiu.
include("lista1_ex1.jl")

x = [0.8 0.3; 0.0 0.5; 0.2 0.1; 0.1 0.9; 0.6 0.5; 0.8 0.1];
y_nat = [4.7 0 0 0 4.5 0];
y_kar = [4.0 4.9 0 4.8 4.6 0];
y_eve = [0 4.8 5.0 4.5 4.8 5.0];
y_ali = [0 0 0 4.5 4.3 0];
y_geo = [0 0 0 0 4.8 4.5];
y_abe = [0 0 0 0 4.5 4.2];

tabela = [y_nat; y_kar; y_eve; y_ali; y_geo; y_abe]';

m,n = size(tabela)
coef = zeros(3,2)

X_kar = zeros(4,3)
count_kar = 0
for i in 1:6
  if tabela[i,2] != 0
    count_kar += 1
    X_kar[count_kar,:] = [x[i,:]' tabela[i,2]]
  end
end
coef[:,1] = quad_min(X_kar[:,1:2], X_kar[:,3])

count_eve = 0
X_eve = zeros(5,3)
for i in 1:6
  if tabela[i,3] != 0
    count_eve += 1
    X_eve[count_eve,:] = [x[i,:]' tabela[i,3]]
  end
end
coef[:,2] = quad_min(X_eve[:,1:2], X_eve[:,3])

#Portanto, as notas estimadas são

#Karla
  #Curitiba Zero Grau
  nota_kar1 = [1 0.2 0.1]*coef[:,1]

  println("Estima-se que Karla pode atribuir a nota
  $nota_kar1 para o filme Curitiba Zero Grau")

  #O Silêncio dos Inocentes
  nota_kar2 = [1 0.8 0.1]*coef[:,1]

  println("Estima-se que Karla pode atribuir a nota
  $nota_kar2 para o filme O Silência dos inocentes")

#Evelin
  #Sem retorno
  nota_eve = [1 0.8 0.1]*coef[:,2]

  println("Estima-se que Evelin pode atribuir a nota
  $nota_eve para o filme Sem Retorno")

#Vetores com as notas de karla e Evelin
Notas_kar1 = [nota_kar1, nota_kar2]
Notas_eve = zeros(6)
Notas_kar = zeros(6)
counter = 0
counterkar = 0
for i = 1:6
  counter += 1
  if tabela[i,2] == 0
    counterkar += 1
    Notas_kar[counter] = Notas_kar1[counterkar][1]
  else
    Notas_kar[i] = tabela[i,2]
  end
end

Notas_eve = tabela[:,3]
Notas_eve[1] = nota_eve[1];
