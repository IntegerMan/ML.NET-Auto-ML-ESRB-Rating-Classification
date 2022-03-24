using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;

namespace MattEland.AI.MLNet.ESRBPredictor.Core;

/// <summary>
/// Manages machine learning models for classifying video games into an ESRB rating given a set of traits about the game's content.
/// </summary>
public class ESRBRatingPredictor
{
    private readonly MLContext _context = new();

    private ITransformer? _model;
    private DataViewSchema? _schema;
    private PredictionEngine<GameInfo, ESRBPrediction>? _predictor;







    /// <summary>
    /// Trains a machine learning model based on ESRB game data in the <paramref name="trainFilePath"/> and <paramref name="validationFilePath"/>.
    /// Once a model is trained, the <see cref="ClassifyGame(GameInfo)">ClassifyGame</see> method can be called to
    /// predict ESRB ratings, or the <see cref="SaveModel(string)">SaveModel</see> method can be called to save the model for future runs.
    ///
    /// See <see href="https://www.kaggle.com/imohtn/video-games-rating-by-esrb">Kaggle</see> for the dataset used for this application.
    /// </summary>
    /// <param name="trainFilePath">The file containing the comma separated values for model training</param>
    /// <param name="validationFilePath">The file containing the comma separated values for model validation</param>
    /// <param name="secondsToTrain">The number of seconds to spend training the machine learning model</param>
    /// <returns>Information about the best model</returns>
    public RunDetail<MulticlassClassificationMetrics> TrainModel(
        string trainFilePath, 
        string validationFilePath, 
        uint secondsToTrain)
    {
        // Load training and test data.
        // See Kaggle dataset at https://www.kaggle.com/imohtn/video-games-rating-by-esrb
        IDataView data = _context.Data.LoadFromTextFile<GameInfo>(
            path: trainFilePath,
            separatorChar: ',',
            hasHeader: true,
            allowQuoting: true,
            trimWhitespace: true);

        // Store the file schema for later. This helps performance slightly
        _schema = data.Schema;

        // NOTE: The code below can be used to verify the state of data after load
        // DataDebuggerPreview? preview = data.Preview();






        // Split our data into two parts - one for training and one for verification
        DataOperationsCatalog.TrainTestData trainTestSplit = _context.Data.TrainTestSplit(data, testFraction: 0.2);








        // Configure the experiment
        MulticlassExperimentSettings settings = new()
        {
            OptimizingMetric = MulticlassClassificationMetric.MacroAccuracy,
            MaxExperimentTimeInSeconds = secondsToTrain,
        };

        MulticlassClassificationExperiment experiment = 
            _context.Auto().CreateMulticlassClassificationExperiment(settings);







        // Actually Train the model
        ExperimentResult<MulticlassClassificationMetrics> result = 
            experiment.Execute(
                trainData: trainTestSplit.TrainSet,
                validationData: trainTestSplit.TestSet,
                labelColumnName: nameof(GameInfo.ESRBRating),
                preFeaturizer: null,
                progressHandler: new MulticlassConsoleProgressReporter());

        // Process our finished result
        _model = result.BestRun.Model;



        // Return the best-performing model. This will include performance metrics
        return result.BestRun;
    }

    /// <summary>
    /// Validates that a trained model exists and creates a PredictionEngine as needed
    /// </summary>
    /// <exception cref="InvalidOperationException">
    /// Thrown if no model has been trained
    /// </exception>
    private void EnsurePredictorExists()
    {
        if (_model == null) throw new InvalidOperationException("You must train or load a model before predicting ESRB ratings");

        _predictor ??= _context.Model.CreatePredictionEngine<GameInfo, ESRBPrediction>(transformer: _model, inputSchema: _schema);
    }



    /// <summary>
    /// Predicts the ESRB ratings of a specified <paramref name="game"/> and returns it
    /// </summary>
    /// <param name="game">The game to classify</param>
    /// <returns>An ESRB prediction including certainty factors</returns>
    /// <exception cref="InvalidOperationException">
    /// Thrown if no model has been trained. Call <see cref="TrainModel(string, string, uint)"/> or <see cref="LoadModel(string)"/> first.
    /// </exception>
    public ESRBPrediction ClassifyGame(GameInfo game)
    {
        EnsurePredictorExists();

        ESRBPrediction prediction = _predictor!.Predict(game);

        return prediction;
    }



    /// <summary>
    /// Saves the model to disk
    /// </summary>
    /// <param name="filename">The path of the model file to save</param>
    /// <exception cref="InvalidOperationException">
    /// Thrown if no model has been trained. Call <see cref="TrainModel(string, string, uint)"/> or <see cref="LoadModel(string)"/> first.
    /// </exception>
    public void SaveModel(string filename)
    {
        if (_model == null) throw new InvalidOperationException("You must train or load a model before saving");

        _context.Model.Save(_model, _schema, filename);
    }

    /// <summary>
    /// Loads the model from disk
    /// </summary>
    /// <param name="filename">The path of the model file to load</param>
    public void LoadModel(string filename)
    {
        _model = _context.Model.Load(filename, out _schema);

        // Whenever our model changes, it's nice to update the prediction engine
        _predictor = _context.Model.CreatePredictionEngine<GameInfo, ESRBPrediction>(transformer: _model, inputSchema: _schema);
    }

}