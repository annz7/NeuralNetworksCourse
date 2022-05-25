using System;

namespace GAN
{
    internal static class Program
    {
        public static void Main(string[] args)
        {
            Console.WriteLine("Loading Data...");
            var xTrain = DataLoader.LoadData(20000, new[] { 0, 1, 2 });
            var gan = new GAN();
            gan.Train(xTrain, 10);
        }
    }
}