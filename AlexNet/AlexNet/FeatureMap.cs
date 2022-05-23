namespace AlexNet
{
    public class FeatureMap
    {
        public readonly double[] Data;
        public readonly double[] Errors;
        public double Bias;
        public double DifferenceBias;

        public FeatureMap(int size)
        {
            Bias = 0.0;
            DifferenceBias = 0.0;
            Data = new double[size];
            Errors = new double[size];
        }
    };
}