# Apply operations on images of the dataset on the given path.

import Augmentor

def augmenting(path,probabilities,size_img,num_of_samples):
  p = Augmentor.Pipeline(path)
  #add greyscale operation
  #p.greyscale(probability=probabilities[0])
  #add random rotation, it can be 90,180,270 or none
  p.rotate_random_90(probability=probabilities[0])
  #add skew operation
  p.skew(probability = probabilities[1])
  #add flip left to the right operation
  p.flip_left_right(probability = probabilities[2])
  #resize the image
  p.resize(probability = probabilities[3], width=size_img[0], height=size_img[1])
  #crop the images
  p.crop_by_size(probability = probabilities[4], width=224, height=224)
  #finally, sample the images
  p.sample(num_of_samples)


#default values to probabilities, size_img and num_of_samples
probabilities = [0.1, 0.3, 0.5, 1.0, 1.0]
size_img = [227, 227]
num_of_samples = 1000

paths = ["Desktop/CM116/Projeto/dataset/catedral_tiradentes_OK",
"Desktop/CM116/Projeto/dataset/fachada_teatro_guaira_OK",
"Desktop/CM116/Projeto/dataset/jardim_botanico_curitiba_estufa_OK",
"Desktop/CM116/Projeto/dataset/memorial_da_cidade_curitiba_OK",
"Desktop/CM116/Projeto/dataset/museu_oscar_niemeyer_OK",
"Desktop/CM116/Projeto/dataset/opera_de_arame_OK",
"Desktop/CM116/Projeto/dataset/parque_tangua_fachada_OK",
"Desktop/CM116/Projeto/dataset/teatro_paiol_OK"]

for path in paths:
    augmenting(path,probabilities,size_img,num_of_samples)
