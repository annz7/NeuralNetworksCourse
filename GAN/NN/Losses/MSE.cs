using System;

namespace NN.Losses
{
    public class MSE : Loss
    {
        public override Matrix Forward(Matrix predicted, Matrix expected)
        {
            var errorSquared = Matrix.ApplyFunction(predicted, expected, (p, l) => Math.Pow(p - l, 2));
            return Matrix.GetFilledMatrix(errorSquared.Average(), errorSquared.Rows, errorSquared.Columns);
        }

        public override Matrix Backward(Matrix predicted, Matrix expected)
        {
            var norm = 2.0 / predicted.Rows;
            return Matrix.ApplyFunction(predicted, expected, (p, l) => norm * (p - l));
        }
    }
}