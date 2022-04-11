using System;
using System.Collections.Generic;

namespace MultilayerPerceptron
{
    public class LossFunctions
    {
        public delegate double LossFunction(double act, double exp);

        public static LossFunction MatchLossFunction(string lossFunction)
        {
            return LossFunctionsDict[lossFunction];
        }
        
        public static LossFunction MatchLossFunctionDerivative(string lossFunction)
        {
            return LossFunctionsDerivative[lossFunction];
        }

        private static readonly Dictionary<string, LossFunctions.LossFunction> LossFunctionsDict =
            new Dictionary<string, LossFunctions.LossFunction>
            {
                { "R^2", R2 },
            };

        private static readonly Dictionary<string, LossFunctions.LossFunction> LossFunctionsDerivative =
            new Dictionary<string, LossFunctions.LossFunction>
            {
                { "R^2", R2Derivative },
            };

        private static double R2(double act, double exp)
        {
            return Math.Pow(exp - act, 2) / 2;
        }

        private static double R2Derivative(double act, double exp)
        {
            return act - exp;
        }
    }
}