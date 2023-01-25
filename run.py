
import sys

sys.path.append('./src')


import ignite.engine
import ignite.contrib.handlers
import ignite.metrics

import json

import src.Trainer as Trainer
import src.Evaluator as Evaluator
import src.Datas as Datas




if __name__ == '__main__' :

    with open(sys.argv[1]) as file:
        config = json.load(file)
        

    train_loader, validation_loader = Datas.get_dataloaders(config)

    ## TRAINER CONFIGURATION

    model, optimizer, criterion, lr_scheduler = Trainer.initialize(config)

    train_step = Trainer.create_train_step(model, optimizer, criterion, lr_scheduler)
    trainer = ignite.engine.Engine(train_step)

    loss_history = []

    trainer.add_event_handler(
        ignite.engine.Events.EPOCH_COMPLETED,
        # Callback
        Trainer.update_loss_history,
        # Parameters of callback
        loss_history
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
    avg_mae = ignite.metrics.RunningAverage(src=mae)

    mae.attach(engine=evaluator, name='mae')
    avg_mae.attach(engine=evaluator, name='avg_mae')

    #### MSE METRICS

    mse = ignite.metrics.MeanSquaredError(output_transform)
    avg_mse = ignite.metrics.RunningAverage(src=mse)

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
        engine=evaluator,
        metric_names=['mae', 'avg_mae', 'mse', 'avg_mse']
    )

    trainer.run(train_loader, max_epochs=config.get('max_epochs', 3))
