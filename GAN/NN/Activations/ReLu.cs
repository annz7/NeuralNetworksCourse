namespace NN.Activations
{
    public class ReLu : Activation
    {
        private readonly double alpha;

        public ReLu(double alpha)
        {
            this.alpha = alpha;
        }

        public override void Forward(Matrix input)
        {
            base.Forward(input);
            Output = Matrix.ApplyFunction(input, i => i > 0 ? i : i * alpha);
        }

        public override void Backward(Matrix gradient)
        {
            InputGradient = Matrix.ApplyFunction(gradient, Output!, (g, o) => g * (o > 0 ? 1 : alpha));
        }
    }
}