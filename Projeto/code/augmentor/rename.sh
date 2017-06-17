paths=("Desktop/CM116/Projeto/dataset/catedral_tiradentes_OK/output"
"Desktop/CM116/Projeto/dataset/fachada_teatro_guaira_OK/output"
"Desktop/CM116/Projeto/dataset/jardim_botanico_curitiba_estufa_OK/output"
"Desktop/CM116/Projeto/dataset/memorial_da_cidade_curitiba_OK/output"
"Desktop/CM116/Projeto/dataset/museu_oscar_niemeyer_OK/output"
"Desktop/CM116/Projeto/dataset/opera_de_arame_OK/output"
"Desktop/CM116/Projeto/dataset/parque_tangua_fachada_OK/output"
"Desktop/CM116/Projeto/dataset/teatro_paiol_OK/output")

#for each path, execute the following command
counter=0
for file in *; do
  mv "$file" "$counter.jpg"
  let counter=counter+1
done
