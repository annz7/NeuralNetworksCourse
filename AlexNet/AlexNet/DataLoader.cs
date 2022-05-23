using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace AlexNet
{
    public static class DataLoader
    {
        private const int Padding = 2;

        public static List<Image> LoadData()
        {
            var path = GetDirPath("AlexNet");
            var trainData = path + "train-images.idx3-ubyte";
            var trainLabels = path + "train-labels.idx1-ubyte";
            var testData = path + "t10k-images.idx3-ubyte";
            var testLabels = path + "t10k-labels.idx1-ubyte";

            var trainImages = Read(trainData, trainLabels);
            var testImages = Read(testData, testLabels);
            return trainImages.Concat(testImages).ToList();
        }

        private static List<Image> Read(string imagesPath, string labelsPath)
        {
            var labels = new BinaryReader(new FileStream(labelsPath, FileMode.Open));
            var images = new BinaryReader(new FileStream(imagesPath, FileMode.Open));

            var magicNumber = images.ReadBigInt32();
            var numberOfImages = images.ReadBigInt32();
            var width = images.ReadBigInt32();
            var height = images.ReadBigInt32();

            var magicLabel = labels.ReadBigInt32();
            var numberOfLabels = labels.ReadBigInt32();

            var imagesList = new List<Image>();

            for (var i = 0; i < numberOfImages; i++)
            {
                var bytes = images.ReadBytes(width * height);
                var arr = new double[height + Padding * 2][];

                for (var j = 0; j < height + Padding * 2; j++)
                {
                    arr[j] = new double[width + Padding * 2];
                }

                for (var j = 0; j < height; j++)
                {
                    for (var k = 0; k < width; k++)
                    {
                        arr[j + Padding][k + Padding] = bytes[j * height + k];
                    }
                }

                imagesList.Add(new Image()
                {
                    Data = arr,
                    Label = labels.ReadByte()
                });
            }

            return imagesList;
        }

        private static int ReadBigInt32(this BinaryReader br)
        {
            var bytes = br.ReadBytes(sizeof(Int32));
            if (BitConverter.IsLittleEndian) Array.Reverse(bytes);
            return BitConverter.ToInt32(bytes, 0);
        }

        private static string GetDirPath(string folderName)
        {
            var path = new DirectoryInfo(AppDomain.CurrentDomain.BaseDirectory).Parent.Parent.Parent;
            var pathToFolder = Path.Combine(path.FullName, "data\\");
            return pathToFolder;
        }
    }
}