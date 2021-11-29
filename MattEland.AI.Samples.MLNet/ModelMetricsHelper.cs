using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MattEland.AI.Samples.MLNet
{
    public static class ModelMetricsHelper
    {
        public static void LogBriefClassificationMetrics(this BinaryClassificationMetrics metrics)
        {
            Console.WriteLine($"Accuracy: {metrics.Accuracy}");
            Console.WriteLine($"F1 Score: {metrics.F1Score}");
            Console.WriteLine($"Area Under Curve (AUC): {metrics.AreaUnderRocCurve}");
            Console.WriteLine();
        }

        public static void LogClassificationMetrics(this BinaryClassificationMetrics metrics)
        {
            Console.WriteLine($"Accuracy: {metrics.Accuracy}");
            Console.WriteLine($"F1 Score: {metrics.F1Score}");
            Console.WriteLine($"Pos. Precision: {metrics.PositivePrecision}");
            Console.WriteLine($"Neg. Precision: {metrics.NegativePrecision}");
            Console.WriteLine($"Pos. Recall: {metrics.PositiveRecall}");
            Console.WriteLine($"Neg. Recall: {metrics.NegativeRecall}");
            Console.WriteLine($"Area Under Curve (AUC): {metrics.AreaUnderRocCurve}");
            Console.WriteLine($"Area Under Precision / Recall Curve: {metrics.AreaUnderPrecisionRecallCurve}");

            Console.WriteLine();
            Console.WriteLine("Confusion Matrix");
            Console.WriteLine();
            Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
            Console.WriteLine();
        }
    }
}
