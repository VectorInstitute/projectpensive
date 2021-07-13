import argparse
from model import CivilCommentsModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--num-train-epochs",
        type=int,
        default=5,
        help="Number of epochs to train for"
    )
    parser.add_argument(
        "-t",
        "--num-train-points",
        default=500000,
        help="Number of data points to grab from training split"
    )
    parsed = parser.parse_args()
    return parsed


if __name__ == "__main__":

    args = parse_args()
    if args.num_train_points == "all":
        model = CivilCommentsModel(
            num_train_epochs=args.num_train_epochs,
        )
    else:
        model = CivilCommentsModel(
            num_train_epochs=args.num_train_epochs,
            num_training_points=int(args.num_train_points)
        )

    model.trainer.train()
    model.trainer.evaluate(eval_dataset=model.test_dataset)
    model.trainer.save_model()

    print("Program training complete")
