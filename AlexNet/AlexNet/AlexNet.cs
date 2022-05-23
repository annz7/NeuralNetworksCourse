using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using static AlexNet.ActivationFunctions;
using static AlexNet.LossFunctions;

namespace AlexNet
{
    public class AlexNet
    {
        private const int BatchSize = 128;
        private const int Width = 32;
        private const int Height = 32;

        private readonly Random rand = new Random();
        private double learningRate;
        private readonly double mu = 0.85;
        private readonly ActivationFunction Activation;
        private readonly ActivationFunction ActivationDerivative;
        private readonly LossFunction Loss;
        private readonly LossFunction LossDerivative;

        private Layer InputLayer;
        private Layer ConvolutionLayer1;
        private Layer PoolingLayer1;
        private Layer ConvolutionLayer2;
        private Layer PoolingLayer2;
        private Layer ConvolutionLayer3;
        private Layer OutputLayer1;
        private Layer OutputLayer2;

        public AlexNet(double learningRate)
        {
            Console.WriteLine("Creating AlexNet...");
            this.learningRate = learningRate;
            Activation = MatchActivationFunction("sigmoid");
            ActivationDerivative = MatchActivationFunctionDerivative("sigmoid");
            Loss = MatchLossFunction("R^2");
            LossDerivative = MatchLossFunctionDerivative("R^2");

            InputLayer = new Layer(0, 1, 0, 0, Width, Height, rand);
            ConvolutionLayer1 = new Layer(1, 6, 5, 5, InputLayer.FeatureMapWidth - 5 + 1,
                InputLayer.FeatureMapHeight - 5 + 1, rand);
            PoolingLayer1 = new Layer(1, 6, 1, 1, ConvolutionLayer1.FeatureMapWidth / 2,
                ConvolutionLayer1.FeatureMapHeight / 2, rand);
            ConvolutionLayer2 = new Layer(6, 16, 5, 5, PoolingLayer1.FeatureMapWidth - 5 + 1,
                PoolingLayer1.FeatureMapHeight - 5 + 1, rand);
            PoolingLayer2 = new Layer(1, 16, 1, 1, ConvolutionLayer2.FeatureMapWidth / 2,
                ConvolutionLayer2.FeatureMapHeight / 2, rand);
            ConvolutionLayer3 = new Layer(16, 120, 5, 5, PoolingLayer2.FeatureMapWidth - 5 + 1,
                PoolingLayer2.FeatureMapHeight - 5 + 1, rand);
            OutputLayer1 = new Layer(120, 84, 1, 1, 1, 1, rand);
            OutputLayer2 = new Layer(84, 10, 1, 1, 1, 1, rand);
        }

        public List<double> Train(List<Image> images, int epochs)
        {
            var lossHistory = new List<double>();

            Console.WriteLine("Training...");

            for (var epoch = 0; epoch < epochs; epoch++)
            {
                Console.WriteLine($"epoch # {epoch + 1}");
                var loss = TrainStep(images);
                lossHistory.Add(loss);
                learningRate *= mu;
            }

            return lossHistory;
        }

        public List<Tuple<byte, double>> Predict(List<double[][]> images)
        {
            Console.WriteLine("Predicting...");
            var predict = new List<Tuple<byte, double>>();

            foreach (var image in images)
            {
                Copy(image, InputLayer.FeatureMaps[0].Data);
                ForwardPropagation(0);
                predict.Add(FindIndex(OutputLayer2));
            }

            return predict;
        }

        private double TrainStep(List<Image> images)
        {
            var shuffledImages = Image.Shuffle(images);
            var batchCount = shuffledImages.Length / BatchSize;

            var progress = 0;
            Console.WriteLine($"progress: {progress}%");

            var loss = new List<double>();

            for (var i = 0; i < batchCount; i++)
            {
                var newProgress = (int)(100.0 * i / (batchCount));

                if (newProgress > progress)
                {
                    progress = newProgress;
                    Console.WriteLine($"progress: {progress}%");
                }

                for (var j = 0; j < BatchSize; j++)
                {
                    var index = i * BatchSize + j;

                    Copy(shuffledImages[index].Data, InputLayer.FeatureMaps[0].Data);

                    loss.Add(ForwardPropagation(shuffledImages[index].Label));
                    BackwardPropagation(shuffledImages[index].Label);
                }

                UpdateWeights();
            }

            return loss.Average();
        }

        #region ForwardPropagation

        private double ForwardPropagation(int label)
        {
            ConvolutionForwardPropagation(InputLayer, ConvolutionLayer1);
            PoolingForwardPropagation(ConvolutionLayer1, PoolingLayer1);
            ConvolutionForwardPropagation(PoolingLayer1, ConvolutionLayer2);
            PoolingForwardPropagation(ConvolutionLayer2, PoolingLayer2);
            ConvolutionForwardPropagation(PoolingLayer2, ConvolutionLayer3);
            FullyConnectedForwardPropagation(ConvolutionLayer3, OutputLayer1);
            FullyConnectedForwardPropagation(OutputLayer1, OutputLayer2);

            return CalculateError(label);
        }

        private void ConvolutionForwardPropagation(Layer previousLayer, Layer layer)
        {
            var size = layer.FeatureMapWidth * layer.FeatureMapHeight;
            for (var i = 0; i < layer.FeatureMapCount; i++)
            {
                for (var j = 0; j < size; j++)
                {
                    layer.FeatureMapTable[j] = 0;
                }

                for (var j = 0; j < previousLayer.FeatureMapCount; j++)
                {
                    var index = j * layer.FeatureMapCount + i;

                    Convolution(
                        previousLayer.FeatureMaps[j].Data, previousLayer.FeatureMapWidth,
                        previousLayer.FeatureMapHeight,
                        layer.Kernels[index].Weights, layer.KernelWidth, layer.KernelHeight,
                        layer.FeatureMapTable, layer.FeatureMapWidth, layer.FeatureMapHeight
                    );
                }

                for (var j = 0; j < size; j++)
                {
                    layer.FeatureMaps[i].Data[j] = Activation(layer.FeatureMapTable[j] + layer.FeatureMaps[i].Bias);
                }
            }
        }

        private static void PoolingForwardPropagation(Layer previousLayer, Layer layer)
        {
            var featureMapWidth = layer.FeatureMapWidth;
            var featureMapHeight = layer.FeatureMapHeight;
            var previousLayerFeatureMapWidth = previousLayer.FeatureMapWidth;
            for (var k = 0; k < layer.FeatureMapCount; k++)
            {
                for (var i = 0; i < featureMapHeight; i++)
                {
                    for (var j = 0; j < featureMapWidth; j++)
                    {
                        var maxValue = previousLayer.FeatureMaps[k].Data[2 * i * previousLayerFeatureMapWidth + 2 * j];
                        for (var n = 2 * i; n < 2 * (i + 1); n++)
                        {
                            for (var m = 2 * j; m < 2 * (j + 1); m++)
                            {
                                maxValue = Math.Max(maxValue,
                                    previousLayer.FeatureMaps[k].Data[n * previousLayerFeatureMapWidth + m]);
                            }
                        }

                        layer.FeatureMaps[k].Data[i * featureMapWidth + j] = maxValue;
                    }
                }
            }
        }

        private void FullyConnectedForwardPropagation(Layer previousLayer, Layer layer)
        {
            for (var i = 0; i < layer.FeatureMapCount; i++)
            {
                var sum = 0.0;
                for (var j = 0; j < previousLayer.FeatureMapCount; j++)
                {
                    sum += previousLayer.FeatureMaps[j].Data[0] *
                           layer.Kernels[j * layer.FeatureMapCount + i].Weights[0];
                }

                sum += layer.FeatureMaps[i].Bias;
                layer.FeatureMaps[i].Data[0] = Activation(sum);
            }
        }

        #endregion

        #region BackwardPropagation

        private void BackwardPropagation(int label)
        {
            for (var i = 0; i < OutputLayer2.FeatureMapCount; i++)
            {
                OutputLayer2.FeatureMaps[i].Errors[0] =
                    LossDerivative(OutputLayer2.FeatureMaps[i].Data[0], ((label == i) ? 1 : 0)) *
                    ActivationDerivative(OutputLayer2.FeatureMaps[i].Data[0]);
            }

            FullyConnectedBackwardPropagation(OutputLayer2, OutputLayer1);
            FullyConnectedBackwardPropagation(OutputLayer1, ConvolutionLayer3);
            ConvolutionBackwardPropagation(ConvolutionLayer3, PoolingLayer2);
            PoolingBackwardPropagation(PoolingLayer2, ConvolutionLayer2);
            ConvolutionBackwardPropagation(ConvolutionLayer2, PoolingLayer1);
            PoolingBackwardPropagation(PoolingLayer1, ConvolutionLayer1);
            ConvolutionBackwardPropagation(ConvolutionLayer1, InputLayer);
        }

        private void FullyConnectedBackwardPropagation(Layer layer, Layer previousLayer)
        {
            for (var i = 0; i < previousLayer.FeatureMapCount; i++)
            {
                previousLayer.FeatureMaps[i].Errors[0] = 0.0;
                for (var j = 0; j < layer.FeatureMapCount; j++)
                {
                    previousLayer.FeatureMaps[i].Errors[0] += layer.FeatureMaps[j].Errors[0] *
                                                              layer.Kernels[i * layer.FeatureMapCount + j].Weights[0];
                }

                previousLayer.FeatureMaps[i].Errors[0] *= ActivationDerivative(previousLayer.FeatureMaps[i].Data[0]);
            }

            for (var i = 0; i < previousLayer.FeatureMapCount; i++)
            {
                for (var j = 0; j < layer.FeatureMapCount; j++)
                {
                    layer.Kernels[i * layer.FeatureMapCount + j].DifferenceWeights[0] +=
                        layer.FeatureMaps[j].Errors[0] * previousLayer.FeatureMaps[i].Data[0];
                }
            }

            for (var i = 0; i < layer.FeatureMapCount; i++)
            {
                layer.FeatureMaps[i].DifferenceBias += layer.FeatureMaps[i].Errors[0];
            }
        }

        private static void PoolingBackwardPropagation(Layer layer, Layer previousLayer)
        {
            var featureMapWidth = layer.FeatureMapWidth;
            var featureMapHeight = layer.FeatureMapHeight;
            var previousLayerFeatureMapWidth = previousLayer.FeatureMapWidth;

            for (var k = 0; k < layer.FeatureMapCount; k++)
            {
                for (var i = 0; i < featureMapHeight; i++)
                {
                    for (var j = 0; j < featureMapWidth; j++)
                    {
                        int row = 2 * i, col = 2 * j;
                        var maxValue = previousLayer.FeatureMaps[k].Data[row * previousLayerFeatureMapWidth + col];
                        for (var n = 2 * i; n < 2 * (i + 1); n++)
                        {
                            for (var m = 2 * j; m < 2 * (j + 1); m++)
                            {
                                if (previousLayer.FeatureMaps[k].Data[n * previousLayerFeatureMapWidth + m] > maxValue)
                                {
                                    row = n;
                                    col = m;
                                    maxValue = previousLayer.FeatureMaps[k].Data[n * previousLayerFeatureMapWidth + m];
                                }
                                else
                                {
                                    previousLayer.FeatureMaps[k].Errors[n * previousLayerFeatureMapWidth + m] = 0.0;
                                }
                            }
                        }

                        previousLayer.FeatureMaps[k].Errors[row * previousLayerFeatureMapWidth + col] =
                            layer.FeatureMaps[k].Errors[i * featureMapWidth + j];
                    }
                }
            }
        }

        private void ConvolutionBackwardPropagation(Layer layer, Layer previousLayer)
        {
            var index = 0;
            var size = previousLayer.FeatureMapWidth * previousLayer.FeatureMapHeight;

            for (var i = 0; i < previousLayer.FeatureMapCount; i++)
            {
                for (var j = 0; j < size; ++j)
                    previousLayer.FeatureMapTable[j] = 0;

                for (var j = 0; j < layer.FeatureMapCount; j++)
                {
                    index = i * layer.FeatureMapCount + j;

                    for (var n = 0; n < layer.FeatureMapHeight; n++)
                    {
                        for (var m = 0; m < layer.FeatureMapWidth; m++)
                        {
                            var error = layer.FeatureMaps[j].Errors[n * layer.FeatureMapWidth + m];
                            for (var ky = 0; ky < layer.KernelHeight; ky++)
                            for (var kx = 0; kx < layer.KernelWidth; kx++)
                                previousLayer.FeatureMapTable[(n + ky) * previousLayer.FeatureMapWidth + m + kx] +=
                                    error * layer.Kernels[index].Weights[ky * layer.KernelWidth + kx];
                        }
                    }
                }

                for (var k = 0; k < size; k++)
                    previousLayer.FeatureMaps[i].Errors[k] =
                        previousLayer.FeatureMapTable[k] * ActivationDerivative(previousLayer.FeatureMaps[i].Data[k]);
            }

            for (var i = 0; i < previousLayer.FeatureMapCount; i++)
            {
                for (var j = 0; j < layer.FeatureMapCount; j++)
                {
                    index = i * layer.FeatureMapCount + j;

                    Convolution(
                        previousLayer.FeatureMaps[i].Data, previousLayer.FeatureMapWidth,
                        previousLayer.FeatureMapHeight,
                        layer.FeatureMaps[j].Errors, layer.FeatureMapWidth, layer.FeatureMapHeight,
                        layer.Kernels[index].DifferenceWeights, layer.KernelWidth, layer.KernelHeight);
                }
            }

            size = layer.FeatureMapWidth * layer.FeatureMapHeight;
            for (var i = 0; i < layer.FeatureMapCount; i++)
            {
                var sum = 0.0;
                for (var k = 0; k < size; k++)
                    sum += layer.FeatureMaps[i].Errors[k];

                layer.FeatureMaps[i].DifferenceBias += sum;
            }
        }

        private static void Convolution(double[] inputData, int inputWidth, int inputHeight, double[] kernel,
            int kernelWidth, int kernelHeight, double[] outputData, int outputWidth, int outputHeight)
        {
            for (var i = 0; i < outputHeight; i++)
            {
                for (var j = 0; j < outputWidth; j++)
                {
                    var sum = 0.0;
                    for (var n = 0; n < kernelHeight; n++)
                    {
                        for (var m = 0; m < kernelWidth; m++)
                        {
                            sum += inputData[(i + n) * inputWidth + j + m] * kernel[n * kernelWidth + m];
                        }
                    }

                    outputData[i * outputWidth + j] += sum;
                }
            }
        }

        #endregion

        #region Update

        private void UpdateWeights()
        {
            UpdateParams(ConvolutionLayer1);
            UpdateParams(PoolingLayer1);
            UpdateParams(ConvolutionLayer2);
            UpdateParams(PoolingLayer2);
            UpdateParams(ConvolutionLayer3);
            UpdateParams(OutputLayer1);
            UpdateParams(OutputLayer2);
        }

        private void UpdateParams(Layer layer)
        {
            var size = layer.KernelWidth * layer.KernelHeight;
            for (var i = 0; i < layer.KernelCount; i++)
            {
                for (var k = 0; k < size; k++)
                {
                    layer.Kernels[i].Weights[k] = GradientDescent(layer.Kernels[i].Weights[k],
                        layer.Kernels[i].DifferenceWeights[k] / BatchSize);
                }
            }

            for (var i = 0; i < layer.FeatureMapCount; i++)
            {
                layer.FeatureMaps[i].Bias = GradientDescent(layer.FeatureMaps[i].Bias,
                    layer.FeatureMaps[i].DifferenceBias / BatchSize);
            }
        }

        private double GradientDescent(double weight, double differenceWeight)
        {
            return weight - learningRate * differenceWeight;
        }

        #endregion

        private double CalculateError(int label)
        {
            var sum = 0.0;
            for (var i = 0; i < OutputLayer2.FeatureMapCount; i++)
            {
                sum += Loss(OutputLayer2.FeatureMaps[i].Data[0], ((label == i) ? 1 : 0)) *
                       ActivationDerivative(OutputLayer2.FeatureMaps[i].Data[0]);
            }

            return sum / OutputLayer2.FeatureMapCount;
        }

        private static void Copy(double[][] src, double[] dst)
        {
            var pointer = 0;
            for (var i = 0; i < src.Length; i++)
            {
                for (var j = 0; j < src[i].Length; j++)
                {
                    dst[pointer] = src[i][j];
                    pointer++;
                }
            }
        }

        private static Tuple<byte, double> FindIndex(Layer layer)
        {
            byte index = 0;
            var sum = layer.FeatureMaps[0].Data[0];
            var maxVal = layer.FeatureMaps[0].Data[0];
            for (byte i = 1; i < layer.FeatureMapCount; i++)
            {
                if (layer.FeatureMaps[i].Data[0] > maxVal)
                {
                    maxVal = layer.FeatureMaps[i].Data[0];
                    index = i;
                }

                if (layer.FeatureMaps[i].Data[0] > 0)
                    sum += layer.FeatureMaps[i].Data[0];
            }

            return Tuple.Create(index, maxVal / sum);
        }
    }
}