using System;

namespace NN.Activations
{
    public class Tanh : Activation
    {
        public override void Forward(Matrix input)
        {
            base.Forward(input);
            Output = Matrix.ApplyFunction(input, Math.Tanh);
        }

        public override void Backward(Matrix gradient)
        {
            InputGradient = Matrix.ApplyFunction(gradient, Output!, (g, o) => g * (1 - Math.Pow(o, 2)));
        }
    }
}