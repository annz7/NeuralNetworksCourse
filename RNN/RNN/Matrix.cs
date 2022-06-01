using System.Collections.Generic;

namespace RNN
{
    public class Matrix
    {
        public int Rows { get; }
        public int Columns { get; }
        private double[,] Data;

        public Matrix(int rows, int columns)
        {
            Rows = rows;
            Columns = columns;
            Data = new double[Rows, Columns];
        }

        public double this[int i, int j]
        {
            get => Data[i, j];
            set => Data[i, j] = value;
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

        public static Matrix GetFilledMatrix(double value, int rows, int columns)
        {
            var result = new Matrix(rows, columns);
            for (var i = 0; i < rows; i++)
            {
                for (var j = 0; j < columns; j++)
                {
                    result[i, j] = value;
                }
            }

            return result;
        }

        public static Matrix GetMatrixFromData(List<double> x)
        {
            var result = new Matrix(x.Count, 1);
            for (var i = 0; i < x.Count; i++)
            {
                result[i, 0] = x[i];
            }

            return result;
        }

        public static Matrix operator +(Matrix first, Matrix second)
        {
            var result = new Matrix(first.Rows, first.Columns);
            for (var i = 0; i < first.Rows; i++)
            {
                for (var j = 0; j < first.Columns; j++)
                {
                    result[i, j] = first[i, j] + second[i, j];
                }
            }

            return result;
        }

        public static Matrix operator -(double first, Matrix second)
        {
            var result = new Matrix(second.Rows, second.Columns);
            for (var i = 0; i < second.Rows; i++)
            {
                for (var j = 0; j < second.Columns; j++)
                {
                    result[i, j] = first - second[i, j];
                }
            }

            return result;
        }

        public static Matrix operator *(Matrix first, Matrix second)
        {
            var result = new Matrix(first.Rows, second.Columns);
            for (var i = 0; i < first.Rows; i++)
            {
                for (var j = 0; j < second.Columns; j++)
                {
                    result[i, j] = 0;
                    for (var k = 0; k < first.Columns; k++)
                    {
                        result[i, j] += first[i, k] * second[k, j];
                    }
                }
            }

            return result;
        }
        public static Matrix operator *(double first, Matrix second)
        {
            var result = new Matrix(second.Rows, second.Columns);
            for (var i = 0; i < second.Rows; i++)
            {
                for (var j = 0; j < second.Columns; j++)
                {
                    result[i, j] = first * second[i, j];
                }
            }

            return result;
        }

        public static Matrix operator -(Matrix first, Matrix second)
        {
            var result = new Matrix(second.Rows, second.Columns);
            for (var i = 0; i < second.Rows; i++)
            {
                for (var j = 0; j < second.Columns; j++)
                {
                    result[i, j] = first[i, j] - second[i, j];
                }
            }

            return result;
        }

        public static Matrix Copy(Matrix matrix)
        {
            var result = new Matrix(matrix.Rows, matrix.Columns);
            for (var i = 0; i < matrix.Rows; i++)
            {
                for (var j = 0; j < matrix.Columns; j++)
                {
                    result[i, j] = matrix[i, j];
                }
            }

            return result;
        }
    }
}