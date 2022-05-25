using System;

namespace NN.Layers
{
    public abstract class Layer
    {
        public Guid Guid { get; } = Guid.NewGuid();
        protected Matrix Input { get; private set; }
        public Matrix Output { get; protected set; }
        public Matrix Parameters { get; set; }
        public Matrix InputGradient { get; set; }
        public Matrix Gradients { get; protected set; }

        public virtual void Forward(Matrix input)
        {
            Input = input;
        }

        public abstract void Backward(Matrix gradient);
    }
}