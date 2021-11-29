using MattEland.AI.Samples.MLNet;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;

public class BinaryExperimentProgressHandler : IProgress<RunDetail<BinaryClassificationMetrics>>
{
    public void Report(RunDetail<BinaryClassificationMetrics> value)
    {
        Console.WriteLine($"{value.TrainerName} in {value.RuntimeInSeconds} seconds");
        Console.WriteLine();

        value.ValidationMetrics.LogBriefClassificationMetrics();
    }
}