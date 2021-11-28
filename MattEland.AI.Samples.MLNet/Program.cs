using Microsoft.ML;
using Microsoft.ML.AutoML;

Console.WriteLine("Hello, World!");

MLContext mlContext = new MLContext();
IDataView trainDataView = mlContext.Data.LoadFromTextFile<Movie>("traintest.csv", hasHeader: true);

var experimentSettings = new BinaryExperimentSettings();

experimentSettings.MaxExperimentTimeInSeconds = 3600;
experimentSettings.OptimizingMetric = BinaryClassificationMetric.F1Score;

var experiment = mlContext.Auto().CreateBinaryClassificationExperiment(experimentSettings);
var result = experiment.Execute(trainDataView, labelColumnName: nameof(Movie.IsChristmasMovie));

Console.WriteLine("Best F1 Score: " + result.BestRun.ValidationMetrics.F1Score);