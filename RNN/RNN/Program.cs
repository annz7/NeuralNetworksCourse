using System;
using System.IO;
using System.Linq;

namespace RNN
{
    internal static class Program
    {
        public static void Main(string[] args)
        {
            System.Threading.Thread.CurrentThread.CurrentCulture = new System.Globalization.CultureInfo("en-US");
            var (xTrain, xTest) = DataLoader.Load(10000, 500);
            xTrain = Scale(xTrain);
            xTest = Scale(xTest);
            var model = new RNN();
            model.Train(xTrain, 10);
            var (test, pred) = model.Predict(xTest);
            var fileWriter = new StreamWriter("Results.txt");
            fileWriter.AutoFlush = true;

            for (var i = 0; i < pred.Count; i++)
            {
                fileWriter.WriteLine(test[i] + " " + pred[i]);
            }

            fileWriter.Close();
        }

        private static double[] Scale(double[] data)
        {
            var mu = data.Average();
            var sigma = data.Sum(t => Math.Pow((t - mu), 2)) / data.Length;
            for (var i = 0; i < data.Length; i++)
            {
                data[i] = (data[i] - mu) / Math.Sqrt(sigma + 1e-08);
            }

            return data;
        }
    }
}