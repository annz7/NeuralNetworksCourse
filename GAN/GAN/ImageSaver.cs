using System;
using System.IO;
using NN;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace GAN
{
    public static class ImageSaver
    {
        public static void Save(Matrix rowData, string filename)
        {
            var data = TransformToMatrixFormat(rowData);
            var width = data.GetLength(0);
            var height = data.GetLength(1);

            using var image = new Image<Rgba32>(width, height);
            for (var x = 0; x < width; x++)
            {
                for (var y = 0; y < height; y++)
                {
                    var floatData = (float)data[x, y];
                    image[x, y] = new Rgba32(floatData, floatData, floatData);
                }
            }

            image.Save(filename);
        }

        public static string CreateImageFolder()
        {
            var imageFolder = DateTime.Now.ToString("MM-dd-HH");
            if (!Directory.Exists(imageFolder))
            {
                Directory.CreateDirectory(imageFolder);
            }

            return imageFolder;
        }

        private static double[,] TransformToMatrixFormat(Matrix rowImages)
        {
            var imagesCount = rowImages.Count;
            var columns = (int)Math.Ceiling(Math.Sqrt(rowImages.Rows));
            var rows = (int)Math.Ceiling((double)rowImages.Rows / columns);

            var width = 30 * columns;
            var height = 30 * rows;
            var matrixImages = new double[width, height];

            for (var x = 0; x < columns; x++)
            {
                for (var y = 0; y < rows; y++)
                {
                    var i0 = 30 * x + 1;
                    var j0 = 30 * y + 1;
                    var image = x * rows + y < imagesCount
                        ? rowImages.GetSubMatrix(x * rows + y, 1)
                        : new Matrix(1, 28 * 28);

                    for (var i = 0; i < 28; i++)
                    {
                        for (var j = 0; j < 28; j++)
                        {
                            matrixImages[i0 + i, j0 + j] = image[j * 28 + i];
                        }
                    }
                }
            }

            return matrixImages;
        }
    }
}