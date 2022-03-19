using Microsoft.ML.AutoML;
using Microsoft.ML.Data;

namespace MattEland.AI.MLNet.ESRBPredictor.Core
{
    public class MulticlassConsoleProgressReporter 
        : IProgress<RunDetail<MulticlassClassificationMetrics>>
    {
        public void Report(RunDetail<MulticlassClassificationMetrics> value)
        {
            if (value.ValidationMetrics != null)
            {
                double accuracy = value.ValidationMetrics.MacroAccuracy;

                Console.WriteLine($"{value.TrainerName} ran in {value.RuntimeInSeconds:0.00} seconds with accuracy of {accuracy:p}");
            }
            else
            {
                Console.WriteLine($"{value.TrainerName} ran in {value.RuntimeInSeconds:0.00} seconds but did not complete. Time likely expired.");
            }
        }
    }
}