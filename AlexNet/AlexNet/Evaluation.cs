using System.Collections.Generic;

namespace AlexNet
{
    public class Evaluation
    {
        public double[,] ConfusionMatrix;

        public Evaluation()
        {
            ConfusionMatrix = new double[11, 11];
        }

        public double CalculateAccuracy()
        {
            var success = 0.0;
            for (var i = 0; i < 10; i++)
            {
                for (var j = 0; j < 10; j++)
                {
                    if (i == j)
                    {
                        success += ConfusionMatrix[i, j];
                    }
                }
            }

            return success / ConfusionMatrix[10, 10];
        }

        public Dictionary<int, double> CalculatePrecision()
        {
            var dict = new Dictionary<int, double>();
            for (var i = 0; i < 10; i++)
            {
                for (var j = 0; j < 10; j++)
                {
                    if (i == j)
                    {
                        dict[i] = ConfusionMatrix[i, j] / ConfusionMatrix[i, 10];
                    }
                }
            }

            return dict;
        }

        public Dictionary<int, double> CalculateRecall()
        {
            var dict = new Dictionary<int, double>();
            for (var i = 0; i < 11; i++)
            {
                for (var j = 0; j < 11; j++)
                {
                    if (i == j)
                    {
                        dict[i] = ConfusionMatrix[i, j] / ConfusionMatrix[10, j];
                    }
                }
            }

            return dict;
        }
    }
}