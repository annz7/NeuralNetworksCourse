using System;
using System.IO;
using NN;

namespace GAN
{
    public class GAN
    {
        private const int NoiseCount = 100;
        private const int BatchSize = 256;
        private readonly Generator generator;
        private readonly Discriminator discriminator;
        private readonly string imageFolder;

        public GAN()
        {
            Console.WriteLine("Creating GAN...");
            generator = new Generator(NoiseCount);
            discriminator = new Discriminator();
            imageFolder = ImageSaver.CreateImageFolder();
        }

        public void Train(Matrix xTrain, int epochs = 1)
        {
            Console.WriteLine("Training...");
            var batchCount = (int)Math.Ceiling(xTrain.Rows / (double)BatchSize);

            for (var epoch = 1; epoch <= epochs; epoch++)
            {
                Console.WriteLine($"Starting Epoch {epoch}");
                xTrain.ShuffleRows();

                for (var step = 1; step <= batchCount; step++)
                {
                    Console.WriteLine($"Step {step} of {batchCount}.");
                    discriminator.Train(generator, xTrain, BatchSize, step);
                    generator.Train(discriminator, BatchSize);
                }

                GenerateAndSaveImages(10, 10, Path.Combine(imageFolder, $"epoch{epoch}.png"));
            }
        }

        private void GenerateAndSaveImages(int cols, int rows, string filename)
        {
            var (createdImages, _) = generator.GetFakeImages(rows * cols);
            ImageSaver.Save(createdImages, filename);
        }
    }
}