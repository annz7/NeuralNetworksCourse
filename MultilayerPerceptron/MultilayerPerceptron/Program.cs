using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ZedGraph;

namespace MultilayerPerceptron
{
    class Program
    {
        static void Main(string[] args)
        {
            System.Threading.Thread.CurrentThread.CurrentCulture = new System.Globalization.CultureInfo("en-US");
            var xTrain = Linspace(-2 * Math.PI, 2 * Math.PI, 1000);
            var yTrain = xTrain.Select(Math.Cos).ToList();
            var xTest = Linspace(2 * Math.PI, 4 * Math.PI, 99);
            var yTest = xTest.Select(Math.Cos).ToList();
            var newXTest = Preprocessing(xTest);

            var model = new Perceptron(new[] { 1, 15, 10, 1 }, "tanh", 0.01);
            var loss = model.Train(xTrain, yTrain, 300);
            var yPred = model.Predict(Preprocessing(xTest));

            //Eval(xTrain, yTrain, yPred);
            WriteToFileAsync(xTest, yTest, yPred, loss);
        }

        private static async Task WriteToFileAsync(List<double> x, List<double> y, List<double> pred, double[] loss)
        {
            const string path = "data\\content.txt";
            //var text = string.Join(" ", x) + "\n" + string.Join(" ", y) + "\n" + string.Join(" ", pred);
            System.Console.WriteLine("x = [" + string.Join(", ", x) + "]");
            System.Console.WriteLine("y = [" + string.Join(", ", y) + "]");
            System.Console.WriteLine("pred = [" + string.Join(", ", pred) + "]");
            System.Console.WriteLine("loss = [" + string.Join(", ", loss) + "]");
        }

        private static void Eval(List<double> xTest, List<double> yTest, List<double> yPred)
        {
            for (var i = 0; i < yTest.Count; i++)
            {
                Console.WriteLine("" + xTest[i] + " -> " + yPred[i] + " exp: " + yTest[i]);
            }
        }

        private static List<double> Linspace(double from, double to, int count)
        {
            var x = new double[count];
            var step = (to - from) / count;

            for (var i = 0; i < count; i++)
            {
                x[i] = from + i * step;
            }

            return x.ToList();
        }

        private static List<double> Preprocessing(List<double> x)
        {
            return (from t in x
                    let isPositive = t > 0 ? 1 : -1
                    let n = (int)(t * isPositive / (2 * Math.PI))
                    select t - isPositive * n * 2 * Math.PI)
                .ToList();
        }
    }
}

