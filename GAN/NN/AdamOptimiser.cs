using System;
using System.Collections.Generic;
using NN.Layers;

namespace NN
{
    public class AdamOptimiser
    {
        private readonly Dictionary<string, Matrix> mt;
        private readonly Dictionary<string, Matrix> vt;
        private long iteration;
        private double Beta1 { get; }
        private double Beta2 { get; }
        private double LearningRate { get; }

        public AdamOptimiser(double learningRate = 2e-4, double beta1 = 0.5, double beta2 = 0.999)
        {
            LearningRate = learningRate;
            Beta1 = beta1;
            Beta2 = beta2;
            mt = new Dictionary<string, Matrix>();
            vt = new Dictionary<string, Matrix>();
            iteration = 0;
        }

        public void Update(DenseLayer denseLayer)
        {
            iteration++;

            var paramFullName = $"{denseLayer.Guid}_weights";
            var weights = denseLayer.Parameters;
            var gradients = denseLayer.Gradients;

            if (!mt.ContainsKey(paramFullName))
            {
                mt[paramFullName] = Matrix.GetFilledMatrix(0, weights.Rows, weights.Columns);
                vt[paramFullName] = Matrix.GetFilledMatrix(0, weights.Rows, weights.Columns);
            }

            mt[paramFullName] = Matrix.ApplyFunction(mt[paramFullName], gradients,
                (m, g) =>
                    Beta1 * m + (1 - Beta1) * g
            );

            vt[paramFullName] = Matrix.ApplyFunction(vt[paramFullName], gradients,
                (v, g) =>
                    Beta2 * v + (1 - Beta2) * Math.Pow(g, 2)
            );

            var learningRateForThisIteration = LearningRate * Math.Sqrt(1 - Math.Pow(Beta2, iteration)) /
                                               (1 - Math.Pow(Beta1, iteration));

            denseLayer.Parameters = Matrix.ApplyFunction(
                weights,
                mt[paramFullName],
                vt[paramFullName],
                (w, m, v) => w - (learningRateForThisIteration * m / (Math.Sqrt(v) + FNN.Epsilon))
            );
        }
    }
}