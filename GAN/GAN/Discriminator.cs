using System;
using NN;
using NN.Activations;
using NN.Layers;
using NN.Losses;

namespace GAN
{
    public class Discriminator
    {
        public readonly FNN Model;

        public Discriminator()
        {
            Model = new FNN(new BinaryCrossEntropy());
            Model.Add(new DenseLayer(28 * 28, 512, new ReLu(0.2)));
            Model.Add(new DenseLayer(512, 256, new ReLu(0.2)));
            Model.Add(new DenseLayer(256, 1, new Sigmoid()));
        }

        public void Train(Generator generator, Matrix xTrain, int batchSize, int step)
        {
            var (realImages, realLabels) = GetRealImages(xTrain, step * batchSize / 2, batchSize / 2);
            var (fakeImages, fakeLabels) = generator.GetFakeImages(batchSize / 2);
            var images = Matrix.Concat(realImages, fakeImages);
            var labels = Matrix.Concat(realLabels, fakeLabels);

            var forward = Model.Forward(images);
            var loss = Model.Loss.Forward(forward, labels);
            Model.Backward(Model.Loss.Backward(forward, labels));
            foreach (var layer in Model.Layers) Model.Optimiser.Update(layer);

            Console.WriteLine($"Discriminator Loss {loss[0]}");
        }

        private static (Matrix images, Matrix labels) GetRealImages(Matrix xTrain, int startRow, int numberOfImages)
        {
            var images = xTrain.GetSubMatrix(startRow, numberOfImages);
            var labels = Matrix.GetFilledMatrix(1.0, numberOfImages, 1);
            return (images, labels);
        }
    }
}