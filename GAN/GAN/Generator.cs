using System;
using NN;
using NN.Activations;
using NN.Layers;
using NN.Losses;

namespace GAN
{
    public class Generator
    {
        private readonly FNN model;
        private readonly int inputCount;

        public Generator(int inputCount)
        {
            this.inputCount = inputCount;
            model = new FNN(new BinaryCrossEntropy());
            model.Add(new DenseLayer(inputCount, 256, new ReLu(0.2)));
            model.Add(new DenseLayer(256, 512, new ReLu(0.2)));
            model.Add(new DenseLayer(512, 1024, new ReLu(0.2)));
            model.Add(new DenseLayer(1024, 28 * 28, new Tanh()));
        }

        public void Train(Discriminator discriminator, int batchSize)
        {
            var noise = GetNoise(batchSize);
            var forgedLabels = Matrix.GetFilledMatrix(1, batchSize, 1);

            var generatorForward = model.Forward(noise);
            var combinedForward = discriminator.Model.Forward(generatorForward);
            var combinedLoss = discriminator.Model.Loss.Forward(combinedForward, forgedLabels);

            discriminator.Model.Backward(discriminator.Model.Loss.Backward(combinedForward, forgedLabels));
            model.Backward(discriminator.Model.Layers[0].InputGradient!);

            foreach (var layer in model.Layers)
            {
                model.Optimiser.Update(layer);
            }

            Console.WriteLine($"Generator Loss {combinedLoss[0]}");
        }

        public (Matrix images, Matrix labels) GetFakeImages(int numberOfImages)
        {
            var noise = GetNoise(numberOfImages);
            var images = model.Predict(noise);
            var labels = Matrix.GetFilledMatrix(0, numberOfImages, 1);
            return (images, labels);
        }

        private Matrix GetNoise(int numberOfImages)
        {
            return Matrix.GetRandomMatrixNormal(0, 1, numberOfImages, inputCount);
        }
    }
}