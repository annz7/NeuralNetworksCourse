using System;

namespace NN
{
    public class Matrix
    {
        public int Rows { get; set; }
        public int Columns { get; }
        public int Count { get; }
        private double[,] Data { get; set; }

        public Matrix(int rows, int columns)
        {
            Rows = rows;
            Columns = columns;
            Count = rows * columns;
            Data = new double[Rows, Columns];
        }

        public double this[int i, int j]
        {
            get => Data[i, j];
            set => Data[i, j] = value;
        }

        public double this[int i]
        {
            get => Data[i / Columns, i % Columns];
            private set => Data[i / Columns, i % Columns] = value;
        }

        public Matrix MatrixMultiply(Matrix other)
        {
            var result = new Matrix(Rows, other.Columns);
            for (var i = 0; i < Rows; i++)
            {
                for (var j = 0; j < other.Columns; j++)
                {
                    result[i, j] = 0;
                    for (var k = 0; k < Columns; k++)
                    {
                        result[i, j] += this[i, k] * other[k, j];
                    }
                }
            }

            return result;
        }

        public Matrix Transpose()
        {
            var result = new Matrix(Columns, Rows);
            for (var i = 0; i < Rows; i++)
            {
                for (var j = 0; j < Columns; j++)
                {
                    result[j, i] = this[i, j];
                }
            }

            return result;
        }

        public static Matrix Concat(Matrix first, Matrix second)
        {
            var result = new Matrix(first.Rows + second.Rows, first.Columns);
            for (var i = 0; i < first.Rows; i++)
            {
                for (var j = 0; j < first.Columns; j++)
                {
                    result[i, j] = first[i, j];
                }
            }

            for (var i = 0; i < second.Rows; i++)
            {
                for (var j = 0; j < second.Columns; j++)
                {
                    result[i + first.Rows, j] = second[i, j];
                }
            }

            return result;
        }

        public static Matrix GetRandomMatrix(double min, double max, int rows, int columns)
        {
            var result = new Matrix(rows, columns);
            for (var i = 0; i < result.Count; i++)
            {
                result[i] = RandomUtils.GetDouble(min, max);
            }

            return result;
        }
        
        public static Matrix GetRandomMatrixNormal(int mean, int std, int rows, int columns)
        {
            var result = new Matrix(rows, columns);
            for (var i = 0; i < result.Count; i++)
            {
                result[i] = RandomUtils.GetNormalDouble(mean, std);
            }

            return result;
        }

        public static Matrix GetFilledMatrix(double value, int rows, int columns)
        {
            var result = new Matrix(rows, columns);
            for (var i = 0; i < result.Count; i++)
            {
                result[i] = value;
            }

            return result;
        }

        public static Matrix ApplyFunction(Matrix input, Func<double, double> f)
        {
            var result = new Matrix(input.Rows, input.Columns);

            for (var i = 0; i < input.Count; i++)
            {
                result[i] = f(input[i]);
            }

            return result;
        }

        public static Matrix ApplyFunction(Matrix first, Matrix second, Func<double, double, double> f)
        {
            var result = new Matrix(first.Rows, first.Columns);

            for (var i = 0; i < first.Count; i++)
            {
                result[i] = f(first[i], second[i]);
            }

            return result;
        }

        public static Matrix ApplyFunction(Matrix first, Matrix second, Matrix third,
            Func<double, double, double, double> f)
        {
            var result = new Matrix(first.Rows, first.Columns);

            for (var i = 0; i < first.Count; i++)
            {
                result[i] = f(first[i], second[i], third[i]);
            }

            return result;
        }

        public double Average()
        {
            var sum = 0.0;
            for (var i = 0; i < Count; i++)
            {
                sum += this[i];
            }

            return sum / Count;
        }

        public Matrix GetSubMatrix(int startRow, int countOfRows)
        {
            var subMatrix = new Matrix(countOfRows, Columns);
            for (var i = 0; i < countOfRows; i++)
            {
                for (var j = 0; j < Columns; j++)
                {
                    subMatrix[i, j] = Data[i + startRow, j];
                }
            }

            return subMatrix;
        }

        public void ShuffleRows()
        {
            for (var i = Rows - 1; i > 0; i--)
            {
                var swapIndex = RandomUtils.GetInt(0, i + 1);
                if (swapIndex != i)
                {
                    SwapRows(i, swapIndex);
                }
            }
        }

        private void SwapRows(int r1, int r2)
        {
            for (var i = 0; i < Columns; i++)
            {
                (Data[r1, i], Data[r2, i]) = (Data[r2, i], Data[r1, i]);
            }
        }
    }
}