
import pathlib
import torch
import torch.nn
import torch.optim
import torch.optim.lr_scheduler

import ignite.engine


import Unfolding.Unfolding as Unfolding

import os
if os.name == 'nt':
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

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

    # Custom loss
    criterion = lossCustom

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
        if getattr(criterion, '__name__', repr(callable)) == "lossCustom":
            loss: torch.Tensor = criterion(model.all_O, model.all_H, results, artifacts)
        else:
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

def lossCustom(O, H, target, J):
    #alpha = beta = torch.tensor([1 if (i==0) else 0.1 for i in range(O.shape[0])])

    resemblance_loss = torch.abs(O - target.unsqueeze(0)).expand((O.shape[0],-1,-1,-1,-1))
    consistance_loss = torch.abs(H - (J - target).unsqueeze(0)).expand((H.shape[0],-1,-1,-1,-1))

    resemblance_loss = torch.mean(resemblance_loss, dim=(1,2,3,4))
    consistance_loss = torch.mean(consistance_loss, dim=(1,2,3,4))

    #alpha.to(device)
    #beta.to(device)
    #resemblance_loss.to(device)
    #consistance_loss.to(device)

    #print("\ndevice alpha:", alpha.get_device(), ", ", torch.cuda.device(alpha.get_device()))
    #print("device beta:", beta.get_device(), ", ", torch.cuda.device(beta.get_device()))
    #print("device resemblance_loss:", resemblance_loss.get_device(), ", ", torch.cuda.device(resemblance_loss.get_device()))
    #print("device consistance_loss:", consistance_loss.get_device(), ", ", torch.cuda.device(consistance_loss.get_device()))

    resemblance_loss = resemblance_loss * 0.1
    consistance_loss = consistance_loss * 0.
    resemblance_loss[0] *= 10
    consistance_loss[0] *= 10

    loss = torch.mean(resemblance_loss + consistance_loss)

    return loss 


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

def save_model(
    engine: ignite.engine.Engine, 
    model: torch.nn.Module, 
    path: pathlib.Path = pathlib.Path('.')
) -> None:
    torch.save(model.state_dict(), path / 'model.pt')