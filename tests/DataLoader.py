
import pathlib
import sys

# sys.path.append('..')

import matplotlib

sys.path.append('.')


# import torch
import torch.utils.data

import ignite.engine
import ignite.contrib.handlers
import ignite.metrics

import json
import pandas
import matplotlib.pyplot
# import numpy

import Unfolding.Datas as Datas

# import Unfolding.Trainer as Trainer
# import Unfolding.Evaluator as Evaluator
# import Unfolding.Datas as Datas
# import Unfolding.Unfolding as Unfolding



if __name__ == '__main__' :

    with open(sys.argv[1]) as file:
        config: dict = json.load(file)

    output_path = pathlib.Path(config.get('output_path', '.'))
    
    train_loader, validation_loader = Datas.get_dataloaders(config)
    
    train_len = len(train_loader.dataset)
    valid_len = len(validation_loader.dataset)

    print('Train dataloader length:', train_len)
    print('Validation dataloader length:', valid_len)
    print('Total :', valid_len+train_len)

    
