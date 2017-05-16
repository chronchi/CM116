include("lista1_ex2b.jl")


for j = 1:6
  list = []
  for k = 2:7
    if df[j,k] >= 4.5
      list = [list; "$(names(df)[k])"]
    end
  end
  println("Recomende o filme $(df[:Filmes][j]) para $list")
end
