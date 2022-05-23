using System;
using System.Collections.Generic;

namespace AlexNet
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
                { "tanh", Tanh },
                { "sigmoid", Sigmoid },
            };
        
        private static readonly Dictionary<string, ActivationFunction> ActivationFunctionsDerivative =
            new Dictionary<string, ActivationFunction>
            {
                { "tanh", TanhDerivative },
                { "sigmoid", SigmoidDerivative },
            };

        private static double Tanh(double input) => Math.Tanh(input);

        private static double TanhDerivative(double input) => 1 / Math.Pow(Math.Cosh(input), 2);

        private static double Sigmoid(double input) => 1 / (1 + Math.Exp(-input));

        private static double SigmoidDerivative(double input) => Sigmoid(input) * (1 - Sigmoid(input));
    }
}