using System;

namespace NN.Losses
{
    public class BinaryCrossEntropy : Loss
    {
        public override Matrix Forward(Matrix predicted, Matrix expected)
        {
            const double min = FNN.Epsilon;
            const double max = 1 - FNN.Epsilon;

            var clipped = Matrix.ApplyFunction(predicted, p => p < min ? min : p > max ? max : p);

            var output = Matrix
                .ApplyFunction(clipped, expected, (c, l) => (-(l * Math.Log(c) + (1 - l) * Math.Log(1 - c))))
                .Average();

            return Matrix.GetFilledMatrix(output, clipped.Rows, clipped.Columns);
        }

        public override Matrix Backward(Matrix predicted, Matrix expected)
        {
            return Matrix.ApplyFunction(predicted, expected, (p, l) => (p - l) / (p * (1 - p)));
        }
    }
}