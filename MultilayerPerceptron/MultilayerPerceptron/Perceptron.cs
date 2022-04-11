using System;
using System.Collections.Generic;
using System.Linq;
using static MultilayerPerceptron.ActivationFunctions;
using static MultilayerPerceptron.LossFunctions;

namespace MultilayerPerceptron
{
    public class Perceptron
    {
        private List<Layer> layers;
        private readonly ActivationFunction Activation;
        private readonly ActivationFunction ActivationDerivative;
        private readonly LossFunction Loss;
        private readonly LossFunction LossDerivative;
        private double learningRate;
        private readonly int layersNumber;
        private double bestLoss = double.MaxValue;
        private List<Layer> bestLayers;

        public Perceptron(int[] neuronNumbers, string activationFunction = "tanh", double learningRate = 0.1,
            string lossFunction = "R^2")
        {
            this.learningRate = learningRate;
            Activation = MatchActivationFunction(activationFunction);
            ActivationDerivative = MatchActivationFunctionDerivative(activationFunction);
            Loss = MatchLossFunction(lossFunction);
            LossDerivative = MatchLossFunctionDerivative(lossFunction);
            layersNumber = neuronNumbers.Length;
            CreateLayers(neuronNumbers);
            bestLayers = new List<Layer>();
            for (var l = 0; l < layersNumber; l++)
            {
                bestLayers.Add(layers[l].Copy());
            }
        }

        public double[] Train(List<double> x, List<double> y, int epochs = 100)
        {
            return Train(x.Select(t => new List<double> { t }).ToList(), y.Select(t => new List<double> { t }).ToList(),
                epochs);
        }

        public double[] Train(List<List<double>> x, List<double> y, int epochs = 100)
        {
            return Train(x, y.Select(t => new List<double> { t }).ToList(), epochs);
        }

        public double[] Train(List<double> x, List<List<double>> y, int epochs = 100)
        {
            return Train(x.Select(t => new List<double> { t }).ToList(), y, epochs);
        }

        public double[] Train(List<List<double>> x, List<List<double>> y, int epochs = 100)
        {
            var loss = new double[epochs];
            for (var k = 0; k < epochs; k++)
            {
                for (var i = 0; i < x.Count(); i++)
                {
                    ForwardPass(x[i]);
                    loss[k] = CalculateError(y[i]);
                    if (loss[k] < bestLoss)
                    {
                        bestLoss = loss[k];
                        for (var l = 0; l < layersNumber; l++)
                        {
                            bestLayers[l] = layers[l].Copy();
                        }
                    }

                    BackwardsPass(y[i]);
                }
            }

            for (var l = 0; l < layersNumber; l++)
            {
                //layers[l] = bestLayers[l].Copy();
            }

            return loss;
        }

        public List<double> Predict(List<double> inputs)
        {
            return inputs.Select(x => PredictForOne(new List<double> { x })).ToList();
        }

        private double PredictForOne(IReadOnlyList<double> inputs)
        {
            ForwardPass(inputs);
            return layers[layersNumber - 1].Neurons.Select(neuron => neuron.Out > 1? 1.0 : neuron.Out).ToList()[0]; //костыль на 1 выход
        }

        private void CreateLayers(int[] neuronNumbers)
        {
            layers = new List<Layer> { new Layer(neuronNumbers[0], 0, true) };

            for (var i = 1; i < neuronNumbers.Length; i++)
            {
                layers.Add(new Layer(neuronNumbers[i], neuronNumbers[i - 1], i < neuronNumbers.Length - 1));
            }
        }

        private void ForwardPass(IReadOnlyList<double> inputs)
        {
            for (var i = 0; i < inputs.Count(); i++)
            {
                if (layers[0].Neurons[i].IsBias) continue;
                layers[0].Neurons[i].Net = inputs[i];
                layers[0].Neurons[i].Out = inputs[i];
            }

            SetOuts();
        }

        private void BackwardsPass(List<double> y)
        {
            //output layer block
            var dE_dout = layers[layersNumber - 1].Neurons.Select((t, i) => LossDerivative(t.Net, y[i])).ToList();
            //var dE_dout = layers[layersNumber - 1].Neurons.Select((t, i) => LossDerivative(t.Out, y[i])).ToList();
            //var dout_dnet = layers[layersNumber - 1].Neurons.Select(t => ActivationDerivative(t.Net)).ToList();
            var dout_dnet = new List<double>();
            //var dE_net = dE_dout.Select((t, i) => t * dout_dnet[i]).ToList();
            var dE_net = dE_dout.Select((t, i) => t).ToList();
            var dnet_dw =
                layers[layersNumber - 1]
                    .Neurons
                    .Select(t2 => layers[layersNumber - 2].Neurons.Select(t1 => t1.Out).ToList())
                    .ToList();

            var corrections = CalculateCorrections(dE_net, dnet_dw);
            //end output layer block

            //hidden layers block
            int k;
            for (k = layersNumber - 2; k > 0; k--)
            {
                dE_dout = new List<double>();
                for (var i = 0; i < layers[k].Neurons.Count; i++)
                {
                    if (layers[k].Neurons[i].IsBias) continue;
                    var sum = 0.0;
                    for (var j = 0; j < layers[k + 1].Neurons.Count; j++)
                    {
                        if (layers[k + 1].Neurons[j].IsBias) continue;
                        sum += dE_net[j] * layers[k + 1].Neurons[j].Weight[i];
                    }

                    dE_dout.Add(sum);
                }

                UpdateWeights(k + 1, corrections);
                dout_dnet = (from t in layers[k].Neurons where !t.IsBias select ActivationDerivative(t.Net)).ToList();

                dE_net = dE_dout.Select((t, i) => t * dout_dnet[i]).ToList();

                dnet_dw = (from t2 in layers[k].Neurons
                    where !t2.IsBias
                    select layers[k - 1].Neurons.Select(t1 => t1.Out).ToList()).ToList();


                corrections = CalculateCorrections(dE_net, dnet_dw);
            }

            //end hidden layers block4
            UpdateWeights(k + 1, corrections);
        }

        private List<List<double>> CalculateCorrections(List<double> dE_net, List<List<double>> dnet_w)
        {
            var corr = new List<List<double>>();
            for (var i = 0; i < dE_net.Count; i++)
            {
                var t = new List<double>();
                for (var j = 0; j < dnet_w[i].Count; j++)
                {
                    t.Add(dE_net[i] * dnet_w[i][j]);
                }

                corr.Add(t);
            }

            return corr;
        }

        private void UpdateWeights(int layerNumber, List<List<double>> corrections)
        {
            for (var i = 0; i < layers[layerNumber].Neurons.Count; i++)
            {
                for (var j = 0; j < layers[layerNumber].Neurons[i].Weight.Length; j++)
                {
                    layers[layerNumber].Neurons[i].Weight[j] -= learningRate * corrections[i][j];
                }
            }
        }

        private void SetOuts()
        {
            for (var i = 1; i < layersNumber; i++)
            {
                foreach (var neuron in layers[i].Neurons.Where(neuron => !neuron.IsBias))
                {
                    neuron.Net = 0;

                    for (var k = 0; k < layers[i - 1].Neurons.Count; k++)
                    {
                        neuron.Net += neuron.Weight[k] * layers[i - 1].Neurons[k].Out;
                    }

                    neuron.Out = i == layersNumber - 1? neuron.Net : Activation(neuron.Net);
                }
            }
        }

        private double CalculateError(List<double> y)
        {
            return y.Select((t, i) => Loss(layers[layersNumber - 1].Neurons[i].Out, t)).Sum() / y.Count;
        }
    }
}