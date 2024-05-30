import sources.bias_var_reg.src as bvr_src
import os
import sys
import wandb
import torch
from torch.distributed.elastic.multiprocessing.errors import record



class Classifier:
    def __init__(self, trainer):
        raise NotImplementedError

    def classify(self, image_path):
        raise NotImplementedError


class OutcomeClassifier(Classifier):
    def __init__(self, trainer):
        self.trainer = trainer
        # read saved weights here somewhere
        self.trainer.eval()



class LabClassifier(Classifier):
    pass


class NoiseTrainer(bvr_src.trainer.GenericTrainer):
    pass

@record
def main():
    # Parse the command line arguments
    parser, config_parser = bvr_src.args._get_arg_parser()
    args, args_txt = bvr_src.args._parse_args(parser, config_parser)

    if args.grad_anomaly:
        torch.autograd.set_detect_anomaly(True)

    # Set up some distributed training deets
    bvr_src.utils.init_distributed(args)

    # Print the experiment name for the slurm output
    if args.rank == 0:
        if args.wandb:
            if args.resume is not None:
                ckpt = torch.load(args.resume)
                run = wandb.init(
                    project=args.wandb_proj,
                    entity=args.wandb_entity,
                    config=args,
                    resume='must',
                    id=ckpt['wandb_id'],
                )
            else:
                run = wandb.init(project=args.wandb_proj, entity=args.wandb_entity, config=args)
            args.name = run.name

        print('+' * 80)
        print(args.name)
        print(f'World size: {args.world_size}')
        print(sys.argv)
        print('+' * 80)
    
    # Create results dir
    results_dir = os.path.join(args.results_dir, args.name)
    os.makedirs(results_dir, exist_ok=True)

    # Train the models
    trainer = bvr_src.trainer.factory.get_trainer(args.method)(args)
    trainer.fit()

if __name__ == '__main__':
    main()

