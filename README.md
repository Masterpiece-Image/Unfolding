# Unfolding

Ce qu'on a fait :
- résolution du problème dimension  => squeeze car (1, 1, 200, 200)
- Loss custom
- Dataset 400

Introduction des artefacts :
- Par quantifaction (déjà fait)
- Seuillage des valeurs singulière (SVD)
- Image simplifier d'imagerie mais qui se ressemble => pas que des phantom logan

---

Est ce qu'il existe plusieurs types d'artefacts ?
Si oui, pourriez vous nous donner quelques types d'artefacts afin que nous
puissions les rajouter sur les images ?

Est ce que vous pourriez nous avoir des images médicales ?
---

Number of parameters : Get 

Number of convolution:

Ce que je vais :
import => setup.py
dataset 





Vérifier les shapes de l'image à chaque itération pour savoir qd ça diminue.

Entrainement avec plus d'image 400 images.



## 



## __Clonning__
```bash
git clone https://github.com/Masterpiece-Image/Unfolding.git
git clone git@github.com:Masterpiece-Image/Unfolding.git
```

## __Environments__

### __Docker__

```bash
TODO
```

### __Conda__


Installation of dependencies :

#### __With GPU__

Before, you must [know your cuda version installed](https://arnon.dk/check-cuda-installed/) (use `nvcc --version`)
```bash
conda create --name unfolding python=3.10
conda activate unfolding
conda install pandas matplotlib tqdm pytorch torchvision ignite pytorch-cuda="your cuda version or latest" -c pytorch -c nvidia
```



#### __With CPU__

```bash
conda install pandas matplotlib tqdm torchvision ignite pytorch torchvision cpuonly -c pytorch
```

#### __From yaml files__

```bash
TODO
```

### __Pip__

```bash
pip3 install torch torchvision ignite pandas matplotlib tqdm
```

#### __From requirements file__

```bash
pip3 install -r requirements.txt
```


### __Nix__


```nix
# File : shell.nix 
{ pkgs ? import <nixpkgs> {}, ... }:

pkgs.mkShell {

    buildInputs = with pkgs; [

        python310

        #python310Packages.pytorchWithCuda
        python310Packages.pytorchWithoutCuda
        python310Packages.torchvision
        python310Packages.ignite

        #python310Packages.numpy
        #python310Packages.scipy
        #python310Packages.scikitimage
        #python310Packages.matplotlib

        #python310Packages.ipywidgets
        #python310Packages.ipykernel

    ];

}
```
Then
```bash
nix-shell shell.nix
```


## __Running__

### __Configuration file (json)__

For configure a training, you can make with a json configuration file.
For example, the file `example.json` can contain:
```json
{
    "model" : {
        "input_channels" : 1,
        "iterations" : 10,
        "num_features" : 48
    },
    "dataset_path" : "./phantom-datas",
    "train_size" : 0.8,
    "batch_size" : 2,
    "output_path" : "output",
    "shuffle" : true,
    "learning_rate" : 0.001,
    "max_epochs" : 1,
    "device" : "cuda" // or cpu
}
```

### __Commands__

Just run :
```bash
python3 run.py ./examples/train3/config.json
```
