
import torch.nn
import torch.utils.data


import ignite.engine



def create_evaluate_function(model: torch.nn.Module):

    # model, optimizer, criterion, lr_scheduler = initialize(config)
    # Define any evaluation
    def eval_step(engine, batch):
        
        # x, y = batch[0].to(idist.device()), batch[1].to(idist.device())
        # artifact, result = batch[0], batch[1]
        artifacts, results = batch

        model.eval() # model.train(False)
        predictions = model(artifacts)
        
        output = {
            'prediction' : predictions, 
            'result' : results
        }
        
        return output

    return eval_step


def update_history_metrics(
    engine: ignite.engine.Engine, 
    evaluator: ignite.engine.Engine,
    dataloader: torch.utils.data.DataLoader,
    history: dict[str, list],
    mode: str
) -> None:

    evaluator.run(dataloader, max_epochs=1)

    # no_epoch = engine.state.epoch
    
    # metrics = evaluator.state.metrics
    # mae = metrics['mae']
    # avg_mae = metrics['avg_mae']
    # mse = metrics['mse']
    # avg_mse = metrics['avg_mse']
    

    # # Print logs
    # str_print = mode + ' Results - Epoch {} - mae: {:.2f} Avg mae: {:.2f} mse: {:.2f} Avg mse: {:.2f}'
    # print(str_print.format(no_epoch, mae, avg_mae, mse, avg_mse))

    # Update history
    for key in evaluator.state.metrics.keys():
        history[key].append(evaluator.state.metrics[key])