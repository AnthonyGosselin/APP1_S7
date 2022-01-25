import argparse
import cv2
import numpy as np
import yaml

from dnn_framework import Network, FullyConnectedLayer, BatchNormalization, ReLU, CrossEntropyLoss
from mnist import MnistTrainer
from dnn_framework.solution.losses import softmax


LEARNING_RATES = [0.01, 0.005, 0.001, 0.0005, 0.0001]
BATCH_SIZE = [16, 32, 64, 128]

# LEARNING_RATES = [0.01]
# BATCH_SIZE = [16]


def prepare_image_batch(file_path):
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    image_resized = cv2.resize(image, (28, 28), interpolation=cv2.INTER_LINEAR)

    image_out = cv2.bitwise_not(image_resized)
    image_out = image_out.astype(float)

    # Need to flatten image
    image_out = image_out.flatten()

    return image_out


def main():
    parser = argparse.ArgumentParser(description='Train Backbone')
    parser.add_argument('--learning_rate', type=float, help='Choose the learning rate')
    parser.add_argument('--batch_size', type=int, help='Set the batch size for the training')
    parser.add_argument('--epoch_count', type=int, help='Choose the epoch count')
    parser.add_argument('--output_path', type=str, help='Choose the output path')

    parser.add_argument('--checkpoint_path', type=str, help='Choose the output path', default=None)

    parser.add_argument("-p", "--predict", action="store_true", help="Run network in inference mode.")
    parser.add_argument("-l", "--loop", action="store_true", help="Loop training for hyperparameter testing.")

    args = parser.parse_args()

    network = create_network(args.checkpoint_path)

    if args.predict:
        # Predict
        network.eval()
        image_path = "input_images/seven_corner.png"
        image_batch = prepare_image_batch(file_path=image_path)
        output = network.forward(image_batch)
        print("OUTPUT:", np.around(softmax(output), 3))
        print(f"Detected number is {np.argmax(output)}")
    else:
        # Train
        if args.loop:
            losses = {}
            subfolder = 'C:\\GitAPP\\APP1_S7\\output'
            for lr in LEARNING_RATES:
                for bs in BATCH_SIZE:
                    name = f'lr-{lr}_bs-{bs}'
                    save_str = f'{subfolder}\\{name}'
                    trainer = MnistTrainer(network, lr, args.epoch_count, bs, save_str)
                    trainer.train(losses, name)
                    network = create_network(args.checkpoint_path)

            losses_file_name = f'{subfolder}\\configs.yaml'
            with open(losses_file_name, 'w') as outfile:
                yaml.dump(losses, outfile, default_flow_style=None)

        else:
            trainer = MnistTrainer(network, args.learning_rate, args.epoch_count, args.batch_size, args.output_path)
            trainer.train()


def create_network(checkpoint_path):
    alpha = 0.1
    layers = [
        FullyConnectedLayer(input_size=784, output_size=128),
        BatchNormalization(size=128, alpha=alpha),
        ReLU(),
        FullyConnectedLayer(input_size=128, output_size=32),
        BatchNormalization(size=32, alpha=alpha),
        ReLU(),
        FullyConnectedLayer(input_size=32, output_size=10)
        ]
    network = Network(layers)
    if checkpoint_path is not None:
        network.load(checkpoint_path)

    return network


if __name__ == '__main__':
    main()
