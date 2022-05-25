#nullable enable
using System.Collections.Generic;
using NN.Layers;
using NN.Losses;

namespace NN
{
    public class FNN
    {
        public const double Epsilon = 1e-5;
        public readonly AdamOptimiser Optimiser = new();
        public List<DenseLayer> Layers { get; }
        public Loss Loss { get; }

        public FNN(Loss loss)
        {
            Layers = new List<DenseLayer>();
            Loss = loss;
        }

        public void Add(DenseLayer denseLayer)
        {
            Layers.Add(denseLayer);
        }

        public Matrix Forward(Matrix input)
        {
            DenseLayer? previousLayer = null;
            foreach (var layer in Layers)
            {
                layer.Forward(previousLayer == null ? input : previousLayer.Output!);
                previousLayer = layer;
            }

            return previousLayer!.Output!;
        }

        public void Backward(Matrix gradOutput)
        {
            var curGradOutput = gradOutput;
            for (var i = Layers.Count - 1; i >= 0; i--)
            {
                var layer = Layers[i];
                layer.Backward(curGradOutput);
                curGradOutput = layer.InputGradient!;
            }
        }

        public Matrix Predict(Matrix inputs)
        {
            return Forward(inputs);
        }
    }
}