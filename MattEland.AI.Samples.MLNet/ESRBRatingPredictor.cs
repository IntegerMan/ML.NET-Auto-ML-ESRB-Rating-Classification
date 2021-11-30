using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;

public class ESRBRatingPredictor
{
    private readonly MLContext _context = new MLContext();

    private ITransformer? _model;
    private DataViewSchema? _schema;

    public string TrainModel(string trainFilePath, string validationFilePath, uint minutesToTrain)
    {
        // Load data. This was built around the Kaggle dataset at https://www.kaggle.com/imohtn/video-games-rating-by-esrb
        IDataView trainData = _context.Data.LoadFromTextFile<GameRating>(trainFilePath, separatorChar: ',', hasHeader: true, allowQuoting: true);
        IDataView validationData = _context.Data.LoadFromTextFile<GameRating>(validationFilePath, separatorChar: ',', hasHeader: true, allowQuoting: true);

        // Configure the experiment
        MulticlassExperimentSettings settings = new MulticlassExperimentSettings()
        {
            OptimizingMetric = MulticlassClassificationMetric.LogLoss,
            MaxExperimentTimeInSeconds = minutesToTrain * 60,
            CacheDirectoryName = null, // in memory
        };

        MulticlassClassificationExperiment experiment = _context.Auto().CreateMulticlassClassificationExperiment(settings);

        // Actually Train the model
        ExperimentResult<MulticlassClassificationMetrics> result = 
            experiment.Execute(trainData: trainData, 
                               validationData: validationData, 
                               labelColumnName: nameof(GameRating.ESRBRating), 
                               progressHandler: new MulticlassProgressReporter());

        // Process our finished result
        _model = result.BestRun.Model;
        _schema = trainData.Schema;

        Console.WriteLine($"Best algorithm: {result.BestRun.TrainerName}");

        // Return a formatted matrix tor analyzing model performance
        return result.BestRun.ValidationMetrics.ConfusionMatrix.GetFormattedConfusionTable();
    }

    public void ClassifyGames(IEnumerable<GameRating> games)
    {
        if (_model == null) throw new InvalidOperationException("You must train or load a model before predicting ESRB ratings");

        PredictionEngine<GameRating, ESRBPrediction> predictEngine = 
            _context.Model.CreatePredictionEngine<GameRating, ESRBPrediction>(transformer: _model, inputSchema: _schema);

        foreach (GameRating game in games)
        {
            ESRBPrediction prediction = predictEngine.Predict(game);

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