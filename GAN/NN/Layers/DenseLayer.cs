using System;
using NN.Activations;

namespace NN.Layers
{
    public class DenseLayer : Layer
    {
        private Activation Activation { get; }

        public DenseLayer(int inputCount, int outputCount, Activation activation)
        {
            Activation = activation;
            var range = Math.Sqrt(6.0 / (inputCount + outputCount));
            Parameters = Matrix.GetRandomMatrix(-range, range, inputCount, outputCount);
        }

        public override void Forward(Matrix input)
        {
            base.Forward(input);
            Output = input.MatrixMultiply(Parameters);

            if (Activation == null) return;
            Activation.Forward(Output);
            Output = Activation.Output;
        }

        public override void Backward(Matrix gradient)
        {
            if (Activation != null)
            {
                Activation.Backward(gradient);
                gradient = Activation.InputGradient!;
            }

            InputGradient = gradient.MatrixMultiply(Parameters.Transpose());
            Gradients = Input!.Transpose().MatrixMultiply(gradient);
        }
    }
}