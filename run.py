
import pathlib
import sys

import matplotlib

sys.path.append('./src')

import torch
import torch.utils.data

import ignite.engine
import ignite.contrib.handlers
import ignite.metrics

import json
import pandas
import matplotlib.pyplot
import numpy


import src.Trainer as Trainer
import src.Evaluator as Evaluator
import src.Datas as Datas
import src.Unfolding as Unfolding





if __name__ == '__main__' :

    with open(sys.argv[1]) as file:
        config: dict = json.load(file)

    
    train_loader, validation_loader = Datas.get_dataloaders(config)

    ## TRAINER CONFIGURATION

    model, optimizer, criterion, lr_scheduler = Trainer.initialize(config)

    train_step = Trainer.create_train_step(
        model=model, 
        optimizer=optimizer, 
        criterion=criterion, 
        lr_scheduler=lr_scheduler
    )

    trainer = ignite.engine.Engine(train_step)

    loss_history = []

    trainer.add_event_handler(
        ignite.engine.Events.EPOCH_COMPLETED,
        # Callback
        Trainer.update_loss_history,
        # Parameters of callback
        loss_history
    )

    output_path = pathlib.Path(config.get('output_path', '.'))

    trainer.add_event_handler(
        ignite.engine.Events.COMPLETED,
        # Callback
        Trainer.save_model,
        # Parameters of callback
        model,
        output_path
    )
    

    # Add progress bar showing batch loss value adn some metrics
    pbar = ignite.contrib.handlers.ProgressBar(
        persist=True
    )

    pbar.attach(
        engine=trainer, 
        output_transform=lambda output: {'loss': output['loss']}
    )


    ## EVALUATOR CONFIGURATION

    evaluate_function = Evaluator.create_evaluate_function(model)
    evaluator = ignite.engine.Engine(evaluate_function)

    ### METRICS CONFIG

    output_transform = \
        lambda output: (output['prediction'], output['result'])

    #### MAE METRICS

    mae = ignite.metrics.MeanAbsoluteError(output_transform)
    avg_mae = ignite.metrics.RunningAverage(src=mae, epoch_bound=False)

    mae.attach(engine=evaluator, name='mae')
    avg_mae.attach(engine=evaluator, name='avg_mae')

    #### MSE METRICS

    mse = ignite.metrics.MeanSquaredError(output_transform)
    avg_mse = ignite.metrics.RunningAverage(src=mse, epoch_bound=False)

    mse.attach(engine=evaluator, name='mse')
    avg_mse.attach(engine=evaluator, name='avg_mse')

    ### HISTORY CONFIGS
    # For each epoch completed, we keep metrics

    validation_history = {
        'mae' : [],
        'avg_mae' : [],
        'mse' : [],
        'avg_mse' : []
    }

    training_history = {
        'mae' : [],
        'avg_mae' : [],
        'mse' : [],
        'avg_mse' : [],
    }

    

    ## Evaluation on datas using for training
    trainer.add_event_handler(
        ignite.engine.Events.EPOCH_COMPLETED,
        # Callback
        Evaluator.update_history_metrics,
        # Parameters of callback
        evaluator, 
        train_loader, 
        training_history,
        'Training Datas'
    )

    ## Evaluation on datas using for validation
    trainer.add_event_handler(
        ignite.engine.Events.EPOCH_COMPLETED,
        # Callback
        Evaluator.update_history_metrics,
        # Parameters of callback
        evaluator, 
        validation_loader, 
        validation_history,
        'Validation Datas'
    )

    # Add progress bar showing batch loss value and some metrics
    pbar = ignite.contrib.handlers.ProgressBar(
        persist=True
    )

    pbar.attach(
        engine=evaluator
        # metric_names=['mae', 'avg_mae', 'mse', 'avg_mse']
    )

    

    trainer.run(train_loader, max_epochs=config.get('max_epochs', 3))

    df_history_train = pandas.DataFrame(data=training_history)
    df_history_valid = pandas.DataFrame(data=validation_history)

    # print(df_history_train)
    # df_history_train.plot()
    
    # matplotlib.pyplot.clf()

    # df_history_train.plot(xlabel='Epoch', ylabel='Metrics')
    # matplotlib.pyplot.title('Train History')
    # matplotlib.pyplot.savefig(output_path / 'train_history')

    # matplotlib.pyplot.clf()

    # df_history_valid.plot(xlabel='Epoch', ylabel='Metrics')
    # matplotlib.pyplot.title('Validation History')
    # matplotlib.pyplot.savefig(output_path / 'validation_history')

    # matplotlib.pyplot.clf()

    # matplotlib.pyplot.plot(loss_history)
    # matplotlib.pyplot.xlabel('Epoch')
    # matplotlib.pyplot.ylabel('Loss')
    # matplotlib.pyplot.title('Loss History')
    # # matplotlib.pyplot.legend()
    # matplotlib.pyplot.savefig(output_path / 'loss_history')

    # matplotlib.pyplot.clf()

    
    # dataset = Datas.ImageDataset(
    #     pathlib.Path(config['dataset_path'])
    # )
    # dataloader = torch.utils.data.DataLoader(dataset)
    # model = Unfolding.Unfolding(
    #     in_channels=config['model'].get('input_channels', 1),
    #     num_features=config['model'].get('num_features', 48),
    #     iterations=config['model'].get('iterations', 10)
    # )
    # model.load_state_dict(torch.load(str(output_path / 'model.pt')))

    # # matplotlib.pyplot.show()
    # model.eval()
    # for no, (artifact, _) in enumerate(dataloader):
    #     print(artifact.shape)
    #     prediction: torch.Tensor = model(artifact)
    #     matplotlib.pyplot.imsave(
    #         output_path / ('train_image_'+str(no)+'.png'),
    #         prediction.detach().numpy()
    #     )
    
    # print(training_history)
    # print(validation_history)
    
