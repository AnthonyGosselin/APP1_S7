import argparse
import cv2
import numpy as np

from dnn_framework import Network, FullyConnectedLayer, BatchNormalization, ReLU
from mnist import MnistTrainer
from dnn_framework.solution.losses import softmax


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
    parser.add_argument('--learning_rate', type=float, help='Choose the learning rate', required=True)
    parser.add_argument('--batch_size', type=int, help='Set the batch size for the training', required=True)
    parser.add_argument('--epoch_count', type=int, help='Choose the epoch count', required=True)
    parser.add_argument('--output_path', type=str, help='Choose the output path', required=True)

    parser.add_argument('--checkpoint_path', type=str, help='Choose the output path', default=None)

    parser.add_argument("-p", "--predict", action="store_true", help="Run network in inference mode.")

    args = parser.parse_args()

    network = create_network(args.checkpoint_path)

    if args.predict:
        # Predict
        network.eval()
        image_batch = prepare_image_batch(file_path="input_images/four_triangle2.png")
        output = network.forward(image_batch)
        print("OUTPUT:", np.around(softmax(output), 3))
        print(f"Detected number is {np.argmax(output)}")
    else:
        # Train
        trainer = MnistTrainer(network, args.learning_rate, args.epoch_count, args.batch_size, args.output_path)
        trainer.train()


def create_network(checkpoint_path):
    alpha = 0.1
    layers = [
        FullyConnectedLayer(128),
        ]
    network = Network(layers)
    if checkpoint_path is not None:
        network.load(checkpoint_path)

    return network


if __name__ == '__main__':
    main()
