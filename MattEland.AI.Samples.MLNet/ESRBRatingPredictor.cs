using Microsoft.ML;
using Microsoft.ML.AutoML;

public class ESRBRatingPredictor
{
    private readonly MLContext _context = new MLContext();

    private ITransformer? _model;
    private DataViewSchema? _schema;

    public string TrainModel(string filepath, uint minutesToTrain)
    {
        IDataView data = _context.Data.LoadFromTextFile<GameRating>(filepath, separatorChar: ',', hasHeader: true, allowQuoting: true, trimWhitespace: true);

        MulticlassExperimentSettings settings = new MulticlassExperimentSettings()
        {
            OptimizingMetric = MulticlassClassificationMetric.LogLoss,
            MaxExperimentTimeInSeconds = minutesToTrain * 60,
            CacheDirectoryName = null, // in memory
        };

        MulticlassClassificationExperiment experiment = _context.Auto().CreateMulticlassClassificationExperiment(settings);

        ExperimentResult<Microsoft.ML.Data.MulticlassClassificationMetrics> result = 
            experiment.Execute(trainData: data, labelColumnName: nameof(GameRating.ESRBRating), progressHandler: new MulticlassProgressReporter());

        _model = result.BestRun.Model;
        _schema = data.Schema;

        Console.WriteLine($"Best algorithm: {result.BestRun.TrainerName}");

        return result.BestRun.ValidationMetrics.ConfusionMatrix.GetFormattedConfusionTable();
    }

    public void ClassifyGames(IEnumerable<GameRating> games)
    {
        if (_model == null) throw new InvalidOperationException("You must train or load a model before predicting ESRB ratings");

        var predictEngine = _context.Model.CreatePredictionEngine<GameRating, ESRBPrediction>(transformer: _model, inputSchema: _schema);

        foreach (GameRating game in games)
        {
            var prediction = predictEngine.Predict(game);

            Console.WriteLine($"Predicting rating of {prediction.ESRBRating} for \"{game.Title}\" with a confidence score of {prediction.Score.Max():p}");
        }
    }

    public void LoadModel(string filename)
    {
        _model = _context.Model.Load(filename, out _schema);
    }

    public void SaveModel(string filename)
    {
        if (_model == null) throw new InvalidOperationException("You must train or load a model before saving");

        _context.Model.Save(_model, _schema, filename);
    }
}