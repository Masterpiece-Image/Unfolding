
import torch
import torch.nn
import torch.optim
import torch.optim.lr_scheduler

import ignite.engine


import Unfolding


def initialize(config: dict[str, str|int|dict]):

    config_model: dict = config.get('model')

    model = Unfolding.Unfolding(
        in_channels=config_model.get('input_channels', 1),
        num_features=config_model.get('num_features', 48),
        iterations=config_model.get('iterations', 10)
    )

    model.to(device = config.get('device', 'cpu'))
   
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr = config.get('learning_rate', 0.001)
    )

    # MAE
    criterion = torch.nn.L1Loss()

    # # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html
    # le = config["num_iters_per_epoch"]
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=le, gamma=0.9)
    lr_scheduler = None

    return model, optimizer, criterion, lr_scheduler


def create_train_step(
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    criterion,
    lr_scheduler: torch.optim.lr_scheduler.StepLR = None
):

    # model, optimizer, criterion, lr_scheduler = initialize(config)
    # Define any training logic for iteration update
    def train_step(engine, batch):
        
        # x, y = batch[0].to(idist.device()), batch[1].to(idist.device())
        # artifact, result = batch[0], batch[1]
        artifacts, results = batch

        model.train()
        predictions = model(artifacts)
        loss: torch.Tensor = criterion(predictions, results)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if not(lr_scheduler is None):
            lr_scheduler.step()

        output = {
            'prediction' : predictions,
            'result' : results,
            'loss' : loss.item()
        }

        return output

    return train_step


def update_loss_history(engine: ignite.engine.Engine, loss_history: list):
    loss_history.append(engine.state.output['loss'])

def print_logs(engine: ignite.engine.Engine):
    strp = 'Epoch [{}/{}] : Loss {:.2f}'
    print(
        strp.format(
            engine.state.epoch,
            engine.state.epoch_length,
            engine.state.iteration,
            engine.state.output['loss']
        )
    )