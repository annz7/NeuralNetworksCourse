using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;

namespace MultilayerPerceptron
{
    public class Layer
    {
        public List<Neuron> Neurons;
        private bool isBias;

        public Layer(int neuronsNumber, int inputsNumber, bool isBias = false)
        {
            this.isBias = isBias;
            CreateNeurons(neuronsNumber, inputsNumber, isBias);
        }

        private Layer()
        {
        }

        public Layer Copy()
        {
            return new Layer()
            {
                Neurons = Neurons.Select(t => t.Copy()).ToList(),
                isBias = this.isBias
            };
        }

        private void CreateNeurons(int neuronsNumber, int inputsNumber, bool isBias)
        {
            Neurons = new List<Neuron>();

            for (var i = 0; i < neuronsNumber; i++)
            {
                Neurons.Add(new Neuron(inputsNumber == 0 ? 0 : inputsNumber + 1));
            }

            if (isBias)
            {
                Neurons.Add(new Neuron(0, true) { Net = 1, Out = 1 });
            }
        }
    }
}