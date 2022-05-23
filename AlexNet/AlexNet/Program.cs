using System;
using System.Collections.Generic;
using System.Data;
using System.IO;
using System.Linq;

namespace AlexNet
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Loading data...");
            var images = DataLoader.LoadData();
            var testCount = (int)(images.Count() * 0.3);
            var test = images.Take(testCount).ToList();
            var train = images.Skip(testCount).ToList();

            var testImages = new List<double[][]>();
            var testLabels = new List<byte>();
            foreach (var image in test)
            {
                testImages.Add(image.Data);
                testLabels.Add(image.Label);
            }

            var model = new AlexNet(0.0001);
            var lossHistory = model.Train(train, 10);
            var predict = model.Predict(testImages);
            Evaluate(predict, testLabels, lossHistory);
        }

        private static void Evaluate(List<Tuple<byte, double>> predicted, List<byte> expected, List<double> lossHistory)
        {
            Console.WriteLine("Evaluating...");

            var evaluation = new Evaluation();

            for (var i = 0; i < expected.Count; i++)
            {
                evaluation.ConfusionMatrix[expected[i], predicted[i].Item1]++;
                evaluation.ConfusionMatrix[expected[i], 10]++;
                evaluation.ConfusionMatrix[10, predicted[i].Item1]++;
                evaluation.ConfusionMatrix[10, 10]++;
            }

            var accuracy = evaluation.CalculateAccuracy();
            var precision = evaluation.CalculatePrecision();
            var recall = evaluation.CalculateRecall();
            
            var path = new DirectoryInfo(AppDomain.CurrentDomain.BaseDirectory).Parent.Parent.Parent;
            var filePath = path.FullName + "\\Result.txt";
            var fileWriter = new StreamWriter(filePath);
            fileWriter.AutoFlush = true;

            fileWriter.Write($"{accuracy}");

            fileWriter.WriteLine("");
            for (var j = 0; j < 10; j++)
            {
                fileWriter.Write($"{precision[j]} ");
            }

            fileWriter.WriteLine("");
            for (var j = 0; j < 10; j++)
            {
                fileWriter.Write($"{recall[j]} ");
            }

            fileWriter.WriteLine("");
            foreach (var loss in lossHistory)
            {
                fileWriter.Write($"{loss} ");
            }

            fileWriter.WriteLine("");
            for (var j = 0; j < expected.Count(); j++)
            {
                fileWriter.WriteLine($"{expected[j]} {predicted[j].Item2} {predicted[j].Item1}");
            }

            fileWriter.Close();
        }
    }
}