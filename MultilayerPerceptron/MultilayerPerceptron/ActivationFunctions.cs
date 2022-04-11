using System;
using System.Collections.Generic;

namespace MultilayerPerceptron
{
    public static class ActivationFunctions
    {
        public delegate double ActivationFunction(double value);

        public static ActivationFunction MatchActivationFunction(string activationFunction)
        {
            return ActivationFunctionsDict[activationFunction];
        }
        
        public static ActivationFunction MatchActivationFunctionDerivative(string activationFunction)
        {
            return ActivationFunctionsDerivative[activationFunction];
        }

        private static readonly Dictionary<string, ActivationFunction> ActivationFunctionsDict =
            new Dictionary<string, ActivationFunction>
            {
                { "tanh", Math.Tanh },
                { "sigmoid", Sigmoid }
            };
        
        private static readonly Dictionary<string, ActivationFunction> ActivationFunctionsDerivative =
            new Dictionary<string, ActivationFunction>
            {
                { "tanh", TanhDerivative },
            };

        private static double TanhDerivative(double input)
        {
            return 1 - Math.Pow(Math.Tanh(input), 2);
        }

        private static double Sigmoid(double input)
        {
            return 1 / (1 + Math.Exp(-input));
        }
    }
}