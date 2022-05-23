using System;

namespace AlexNet
{
    public class Kernel
    {
        public readonly double[] Weights;
        public readonly double[] DifferenceWeights;

        public Kernel(int size, Random rand)
        {
            Weights = new double[size];
            for (var i = 0; i < size; i++)
            {
                Weights[i] = (rand.NextDouble() - 0.5) * 0.2;
            }

            DifferenceWeights = new double[size];
        }
    };
}