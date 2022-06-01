using System;
using System.IO;
using System.Linq;

namespace RNN
{
    public static class DataLoader
    {
        public static (double[] trainData, double[] testData) Load(int trainCount, int testCount)
        {
            var path = GetDirPath("RNN") + "data.txt";
            var f = new StreamReader(path);
            
            var train = f.ReadLine()?.Trim().Split();
            var trainData = new double[train.Length];
            for (var i = 0; i < train.Length; i++)
                trainData[i] = Convert.ToDouble(train[i]);
            
            var test = f.ReadLine()?.Trim().Split();
            var testData = new double[test.Length];
            for (var i = 0; i < test.Length; i++)
                testData[i] = Convert.ToDouble(test[i]);
            return (trainData.Take(trainCount).ToArray(), trainData.Skip(trainCount).Take(testCount).ToArray());
        }
        
        private static string GetDirPath(string folderName)
        {
            var path = new DirectoryInfo(AppDomain.CurrentDomain.BaseDirectory).Parent.Parent.Parent;
            var pathToFolder = Path.Combine(path.FullName, "data\\");
            return pathToFolder;
        }
    }
}