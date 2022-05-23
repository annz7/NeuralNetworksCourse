using System;
using System.Collections.Generic;

namespace AlexNet
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
                { "log", Log},
            };

        private static readonly Dictionary<string, LossFunctions.LossFunction> LossFunctionsDerivative =
            new Dictionary<string, LossFunctions.LossFunction>
            {
                { "R^2", R2Derivative },
                { "log", LogDerivative},
            };

        private static double R2(double act, double exp)
        {
            return Math.Pow(exp - act, 2) / 2;
        }

        private static double R2Derivative(double act, double exp)
        {
            return act - exp;
        }
        
        private static double Log(double act, double exp)
        {
            if (Math.Abs(exp - 1) < 0.001)
                return - Math.Log(act);
            return - Math.Log(1 - act);
        }

        private static double LogDerivative(double act, double exp)
        {
            if (Math.Abs(exp - 1) < 0.001)
                return - 1 / act;
            return 1 / (1 - act);
        }
    }
}