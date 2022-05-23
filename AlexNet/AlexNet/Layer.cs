using System;

namespace AlexNet
{
    public class Layer
    {
        public readonly int FeatureMapWidth;
        public readonly int FeatureMapHeight;
        public readonly int FeatureMapCount;
        public readonly FeatureMap[] FeatureMaps;

        public readonly int KernelWidth;
        public readonly int KernelHeight;
        public readonly int KernelCount;
        public readonly Kernel[] Kernels;

        public readonly double[] FeatureMapTable;

        public Layer(int previousLayerFeatureMapCount, int featureMapCount, int kernelWidth, int kernelHeight,
            int featureMapWidth, int featureMapHeight, Random rand)
        {
            KernelCount = previousLayerFeatureMapCount * featureMapCount;
            KernelWidth = kernelWidth;
            KernelHeight = kernelHeight;
            Kernels = new Kernel[KernelCount];
            var size = kernelWidth * kernelHeight;
            for (var i = 0; i < previousLayerFeatureMapCount; i++)
            {
                for (var j = 0; j < featureMapCount; j++)
                {
                    Kernels[i * featureMapCount + j] = new Kernel(size, rand);
                }
            }

            FeatureMapCount = featureMapCount;
            FeatureMapWidth = featureMapWidth;
            FeatureMapHeight = featureMapHeight;
            FeatureMaps = new FeatureMap[featureMapCount];
            size = featureMapWidth * featureMapHeight;
            for (var i = 0; i < featureMapCount; i++)
            {
                FeatureMaps[i] = new FeatureMap(size);
            }

            FeatureMapTable = new double[size];
        }
    };
}