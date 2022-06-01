using System;
using System.Collections.Generic;
using System.Linq;

namespace RNN
{
    public class RNN
    {
        private readonly int inputCount = 48;
        private readonly int outputCount = 1;
        private readonly int memoryCount = 50;
        private readonly int layersCount = 15;
        private Matrix Wxh;
        private Matrix Whh;
        private Matrix Why;
        private Matrix dWxh;
        private Matrix dWhh;
        private Matrix dWhy;
        private Matrix dWxh_t;
        private Matrix dWhh_t;
        private Matrix dWhy_t;
        private Matrix dWxh_i;
        private Matrix dWhh_i;
        private readonly double learningRate = 1e-6;
        private readonly int bpttTruncate = 5;
        private readonly int MinCLipValue = -10;
        private readonly int MaxClipValue = 10;
        private List<Matrix> xTrain;
        private List<Matrix> yTrain;
        private List<Dictionary<string, Matrix>> layers;
        private Matrix previousActivation;
        private Matrix outWxh;
        private Matrix outWhh;
        private Matrix outWhy;
        private Matrix net;
        private Matrix activation;

        public RNN()
        {
            Wxh = CreateWeights(memoryCount, inputCount);
            dWxh = Matrix.GetFilledMatrix(0, memoryCount, inputCount);
            dWxh_t = Matrix.GetFilledMatrix(0, memoryCount, inputCount);
            dWxh_i = Matrix.GetFilledMatrix(0, memoryCount, inputCount);

            Whh = CreateWeights(memoryCount, memoryCount);
            dWhh = Matrix.GetFilledMatrix(0, memoryCount, memoryCount);
            dWhh_t = Matrix.GetFilledMatrix(0, memoryCount, memoryCount);
            dWhh_i = Matrix.GetFilledMatrix(0, memoryCount, memoryCount);

            Why = CreateWeights(outputCount, memoryCount);
            dWhy = Matrix.GetFilledMatrix(0, outputCount, memoryCount);
            dWhy_t = Matrix.GetFilledMatrix(0, outputCount, memoryCount);
        }

        public void Train(double[] data, int epochs)
        {
            SetXY(data);

            for (var epoch = 0; epoch < epochs; epoch++)
            {
                var loss = 0.0;

                for (var i = 0; i < yTrain.Count; i++)
                {
                    if (i % 1000 == 0) Console.WriteLine(i);
                    var x = xTrain[i];
                    var y = yTrain[i];

                    layers = new List<Dictionary<string, Matrix>>();
                    previousActivation = Matrix.GetFilledMatrix(0, memoryCount, outputCount);
                    ResetGradients();
                    ForwardPass(x);
                    BackwardPass(x, y);
                }

                for (var i = 0; i < yTrain.Count; i++)
                {
                    var x = xTrain[i];
                    var y = yTrain[i];
                    previousActivation = Matrix.GetFilledMatrix(0, memoryCount, outputCount);
                    Matrix result = null;
                    for (var t = 0; t < layersCount; t++)
                    {
                        var newInput = Matrix.GetFilledMatrix(0, x.Rows, x.Columns);
                        newInput[t, 0] = x[t, 0];
                        net = Wxh * newInput + Whh * previousActivation;
                        activation = Sigmoid(net);
                        result = Why * activation;
                        previousActivation = activation;
                    }

                    var lossCurrent = Math.Abs(y[0, 0] - result[0, 0]);
                    loss += lossCurrent;
                }

                loss /= yTrain.Count;
                Console.WriteLine($"Epoch: {epoch + 1} Loss: {loss}");

                if (loss < 1) return;
            }
        }

        public (List<double>, List<double>) Predict(double[] data)
        {
            var yPred = new List<double>();
            var yTest = new List<double>();
            SetXY(data);
            
            for (var i = 0; i < yTrain.Count; i++)
            {
                var x = xTrain[i];
                var y = yTrain[i];
                previousActivation = Matrix.GetFilledMatrix(0, memoryCount, outputCount);
                for (var t = 0; t < layersCount; t++)
                {
                    var newInput = Matrix.Copy(x);
                    var add = Wxh * newInput + Whh * previousActivation;
                    activation = Sigmoid(add);
                    var result = Why * activation;
                    yPred.Add(result[0, 0]);
                    yTest.Add(y[0, 0]);
                    previousActivation = activation;
                }
            }

            return (yTest, yPred);
        }

        private void SetXY(IReadOnlyCollection<double> data)
        {
            xTrain = new List<Matrix>();
            yTrain = new List<Matrix>();
            for (var i = 0; i < data.Count - inputCount; i++)
            {
                xTrain.Add(Matrix.GetMatrixFromData(data.Skip(i).Take(inputCount).ToList()));
                yTrain.Add(Matrix.GetMatrixFromData(data.Skip(i + inputCount).Take(1).ToList()));
            }
        }

        private static Matrix CreateWeights(int rows, int columns)
        {
            var rand = new Random();
            var matrix = new Matrix(rows, columns);
            for (var i = 0; i < rows; i++)
            {
                for (var j = 0; j < columns; j++)
                {
                    matrix[i, j] = rand.NextDouble();
                }
            }

            return matrix;
        }

        private void ResetGradients()
        {
            dWxh = Matrix.GetFilledMatrix(0, memoryCount, inputCount);
            dWxh_t = Matrix.GetFilledMatrix(0, memoryCount, inputCount);
            dWxh_i = Matrix.GetFilledMatrix(0, memoryCount, inputCount);

            dWhh = Matrix.GetFilledMatrix(0, memoryCount, memoryCount);
            dWhh_t = Matrix.GetFilledMatrix(0, memoryCount, memoryCount);
            dWhh_i = Matrix.GetFilledMatrix(0, memoryCount, memoryCount);
        }

        private void ForwardPass(Matrix x)
        {
            for (var t = 0; t < layersCount; t++)
            {
                var inputs = Matrix.Copy(x);
                outWxh = Wxh * inputs;
                outWhh = Whh * previousActivation;
                net = outWxh + outWhh;
                activation = Sigmoid(net);
                outWhy = Why * activation;
                layers.Add(new Dictionary<string, Matrix> { { "activation", activation }, { "previousActivation", previousActivation } });
                previousActivation = activation;
            }
        }

        private void BackwardPass(Matrix x, Matrix y)
        {
            var dres_Why = outWhy - y;
            for (var t = 0; t < layersCount; t++)
            {
                dWhy_t = dres_Why * layers[t]["activation"].Transpose();
                var dsv = Why.Transpose() * dres_Why;
                var ds = Matrix.Copy(dsv);
                var dnet = net * (1 - net) * ds;
                var dres_Whh = dnet * Matrix.GetFilledMatrix(1, outWhh.Rows, outWhh.Columns);
                var dpreviousActivation = Whh.Transpose() * dres_Whh;

                for (var i = t - 1; i < Math.Max(-1, t - bpttTruncate - 1); i--)
                {
                    ds = dsv + dpreviousActivation;
                    dnet = net * (1 - net) * ds;

                    dres_Whh = dnet * Matrix.GetFilledMatrix(1, outWhh.Rows, outWhh.Columns);
                    var dres_Wxh = dnet * Matrix.GetFilledMatrix(1, outWxh.Rows, outWxh.Columns);

                    dWhh_i = Whh * layers[t]["previousActivation"];
                    dpreviousActivation = Whh.Transpose() * dres_Whh;
                    
                    var newInput = Matrix.Copy(x);
                    dWxh_i = Wxh * newInput;
                    var dx = Wxh.Transpose() * dres_Wxh;

                    dWxh_t += dWxh_i;
                    dWhh_t += dWhh_i;
                }

                dWhy += dWhy_t;
                dWhh += dWhh_t;
                dWxh += dWxh_t;
                CropGradients();
            }

            Wxh -= learningRate * dWxh;
            Whh -= learningRate * dWhh;
            Why -= learningRate * dWhy;
        }

        private static Matrix Sigmoid(Matrix data)
        {
            for (var i = 0; i < data.Rows; i++)
            {
                for (var j = 0; j < data.Columns; j++)
                {
                    data[i, j] = 1 / (1 + Math.Exp(-data[i, j]));
                }
            }

            return data;
        }

        private void CropGradients()
        {
            dWxh = CropValues(dWxh);
            dWhh = CropValues(dWhh);
            dWhy = CropValues(dWhy);
        }

        private Matrix CropValues(Matrix data)
        {
            for (var i = 0; i < data.Rows; i++)
            {
                for (var j = 0; j < data.Columns; j++)
                {
                    if (data[i, j] > MaxClipValue) data[i, j] = MaxClipValue;
                    if (data[i, j] < MinCLipValue) data[i, j] = MinCLipValue;
                }
            }

            return data;
        }
    }
}