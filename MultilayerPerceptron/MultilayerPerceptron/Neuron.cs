using System;
using System.Collections.Generic;

namespace MultilayerPerceptron
{
    public class Neuron
    {
        public double[] Weight { get; set; }
        public double Net { get; set; }
        public double Out { get; set; }
        public bool IsBias { get; set; }

        private Neuron()
        {
        }

        public Neuron(int inputsNumber, bool isBias = false)
        {
            IsBias = isBias;
            var rand = new Random();
            Weight = new double[inputsNumber];

            for (var i = 0; i < inputsNumber; i++)
            {
                //Weight[i] = rand.NextDouble();
                Weight[i] = (double)rand.Next(-50, 50) / 100;
            }
        }

        public Neuron Copy()
        {
            var newNeuron = new Neuron
            {
                Weight = this.Weight,
                Net = this.Net,
                Out = this.Out,
                IsBias = this.IsBias
            };
            return newNeuron;
        }
    }
}