## Environment

### Docker

### Conda

### Pip

### Nix

## Running

### Configuration file (json)

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
    "device" : "gpu"
}
```

### Commands

Just run :
```bash
python3 run.py example.json
```
