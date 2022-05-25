using System;

namespace NN
{
    public static class RandomUtils
    {
        private static readonly Random Rand = new();

        public static double GetDouble(double min, double max)
        {
            return min + Rand.NextDouble() * (max - min);
        }
        
        public static int GetInt(int min, int max)
        {
            return Rand.Next(min, max);
        }

        public static double GetNormalDouble(int mean, int standardDeviation)
        {
            return mean + standardDeviation * GetNormalDouble();
        }

        private static double GetNormalDouble()
        {
            var u1 = Rand.NextDouble();
            var u2 = Rand.NextDouble();
            var r = Math.Sqrt(-2.0 * Math.Log(u1));
            var theta = 2.0 * Math.PI * u2;
            return r * Math.Sin(theta);
        }
    }
}