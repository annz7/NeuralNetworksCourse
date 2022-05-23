using System;
using System.Collections.Generic;

namespace AlexNet
{
    public class Image
    {
        public byte Label { get; set; }
        public double[][] Data { get; set; }

        public static Image[] Shuffle(List<Image> images)
        {
            var rand = new Random();
            var list = images.ToArray();
            var n = list.Length;
            while (n > 1)
            {
                n--;
                var k = rand.Next(n + 1);
                (list[k], list[n]) = (list[n], list[k]);
            }

            return list;
        }
    }
}