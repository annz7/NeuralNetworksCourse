using System;

namespace NN.Activations
{
    public class Sigmoid : Activation
    {
        public override void Forward(Matrix input)
        {
            base.Forward(input);
            var exp = Matrix.ApplyFunction(input, Math.Exp);
            Output = Matrix.ApplyFunction(exp, ex => ex / (1 + ex));
        }

        public override void Backward(Matrix gradient)
        {
            InputGradient = Matrix.ApplyFunction(gradient, Output!, (g, o) => g * o * (1 - o));
        }
    }
}