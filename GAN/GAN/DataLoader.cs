#nullable enable
using System;
using System.IO;
using System.Linq;
using NN;

namespace GAN
{
    public static class DataLoader
    {
        public static Matrix LoadData(int numberOfImages, int[]? filter = null)
        {
            var path = GetDirPath("GAN");
            var trainData = path + "train-images.idx3-ubyte";
            var trainLabels = path + "train-labels.idx1-ubyte";

            return Read(trainData, trainLabels, numberOfImages, filter);
        }

        private static Matrix Read(string imagesPath, string labelsPath, int numberOfImagesNeed, int[]? filter)
        {
            var labels = new BinaryReader(new FileStream(labelsPath, FileMode.Open));
            var magicLabel = labels.ReadBigInt32();
            var numberOfLabels = labels.ReadBigInt32();

            var images = new BinaryReader(new FileStream(imagesPath, FileMode.Open));
            var magicNumber = images.ReadBigInt32();

            var numberOfImages = images.ReadBigInt32();
            var width = images.ReadBigInt32();
            var height = images.ReadBigInt32();

            var imagesList = new Matrix(numberOfImagesNeed, width * height);
            var k = 0;
            for (var i = 0; i < numberOfImages; i++)
            {
                var bytes = images.ReadBytes(width * height);
                if (filter != null && filter.Any() && !filter.Contains(labels.ReadByte())) continue;

                for (var j = 0; j < bytes.Length; j++)
                {
                    imagesList[k, j] = (double)bytes[j] / 255 * 2 - 1;
                }

                k++;
                if (numberOfImagesNeed <= k) break;
            }

            if (k < numberOfImagesNeed)
            {
                imagesList.Rows = k;
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