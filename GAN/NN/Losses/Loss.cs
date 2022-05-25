namespace NN.Losses
{
    public abstract class Loss
    {
        public abstract Matrix Forward(Matrix predicted, Matrix expected);

        public abstract Matrix Backward(Matrix predicted, Matrix expected);
    }
}