using System.Collections.Immutable;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;

namespace MattEland.AI.MLNet.ESRBPredictor.Core 
{

    /// <summary>
    /// Manages machine learning models for classifying video games into an ESRB rating given a set of traits about the game's content.
    /// </summary>
    public class ESRBRatingPredictor
    {
        private readonly MLContext _context = new();

        private ITransformer? _model;
        private DataViewSchema? _schema;
        private PredictionEngine<GameInfo, ESRBPrediction> _predictor;

        /// <summary>
        /// Trains a machine learning model based on ESRB game data in the <paramref name="trainFilePath"/> and <paramref name="validationFilePath"/>.
        /// Once a model is trained, the <see cref="ClassifyGames(IEnumerable{GameInfo})">ClassifyGames</see> method can be called to
        /// predict ESRB ratings, or the <see cref="SaveModel(string)">SaveModel</see> method can be called to save the model for future runs.
        ///
        /// See <see href="https://www.kaggle.com/imohtn/video-games-rating-by-esrb">Kaggle</see> for the dataset used for this application.
        /// </summary>
        /// <param name="trainFilePath">The file containing the comma separated values for model training</param>
        /// <param name="validationFilePath">The file containing the comma separated values for model validation</param>
        /// <param name="secondsToTrain">The number of seconds to spend training the machine learning model</param>
        /// <returns>Information about the best model</returns>
        public RunDetail<MulticlassClassificationMetrics> TrainModel(string trainFilePath, string validationFilePath, uint secondsToTrain)
        {
            // Load data. This was built around the Kaggle dataset at https://www.kaggle.com/imohtn/video-games-rating-by-esrb
            IDataView trainData = _context.Data.LoadFromTextFile<GameInfo>(
                path: trainFilePath,
                separatorChar: ',',
                hasHeader: true,
                allowQuoting: true);

            IDataView validationData = _context.Data.LoadFromTextFile<GameInfo>(
                path: validationFilePath, 
                separatorChar: ',', 
                hasHeader: true, 
                allowQuoting: true);

            // Configure the experiment
            MulticlassExperimentSettings settings = new()
            {
                OptimizingMetric = MulticlassClassificationMetric.LogLoss,
                MaxExperimentTimeInSeconds = secondsToTrain,
            };

            MulticlassClassificationExperiment experiment = _context.Auto().CreateMulticlassClassificationExperiment(settings);

            // Actually Train the model
            ExperimentResult<MulticlassClassificationMetrics> result = experiment.Execute(
                trainData: trainData,
                validationData: validationData,
                labelColumnName: nameof(GameInfo.ESRBRating),
                progressHandler: new MulticlassConsoleProgressReporter());

            // Process our finished result
            _model = result.BestRun.Model;
            _schema = trainData.Schema;

            // Whenever our model changes, it's nice to update the prediction engine
            _predictor = _context.Model.CreatePredictionEngine<GameInfo, ESRBPrediction>(transformer: _model, inputSchema: _schema);

            /* This code is the beginning of working with feature explainability, but is currently bugged for AutoML
             * See https://github.com/dotnet/machinelearning/issues/6084 for bug details and current workaround status
             * See https://docs.microsoft.com/en-us/dotnet/machine-learning/how-to-guides/explain-machine-learning-model-permutation-feature-importance-ml-net for more info on explainability

            ImmutableDictionary<string, MulticlassClassificationMetricsStatistics>? permutationFeatureImportance =
                _context
                    .MulticlassClassification
                    .PermutationFeatureImportance(_model, trainData, permutationCount: 3);

            // The below is a draft that would work with regression, but needs to be adapted to multi-class classification

            var featureImportanceMetrics = permutationFeatureImportance
                        .Select((metric, index) => new { index, metric.RSquared })
                        .OrderByDescending(myFeatures => Math.Abs(myFeatures.RSquared.Mean));

                Console.WriteLine("Feature\tPFI");

                foreach (var feature in featureImportanceMetrics)
                {
                    Console.WriteLine($"{featureColumnNames[feature.index],-20}|\t{feature.RSquared.Mean:F6}");
                }
            */

            // Return a formatted matrix tor analyzing model performance
            return result.BestRun;
        }

        /// <summary>
        /// Predicts the ESRB ratings of a specified <paramref name="game"/> and returns it
        /// </summary>
        /// <param name="game">The game to classify</param>
        /// <returns>An ESRB prediction including certainty factors</returns>
        /// <exception cref="InvalidOperationException">
        /// Thrown if no model has been trained. Call <see cref="TrainModel(string, string, uint)"/> or <see cref="LoadModel(string)"/> first.
        /// </exception>
        public GameClassificationResult ClassifyGame(GameInfo game)
        {
            if (_predictor == null) throw new InvalidOperationException("You must train or load a model before predicting ESRB ratings");

            ESRBPrediction prediction = _predictor.Predict(game);

            return new GameClassificationResult(prediction, game);
        }

        /// <summary>
        /// Predicts the ESRB ratings of incoming <paramref name="games"/> and returns those predictions
        /// </summary>
        /// <param name="games">The games to classify</param>
        /// <returns>A series of ESRB predictions including certainty factors</returns>
        /// <exception cref="InvalidOperationException">
        /// Thrown if no model has been trained. Call <see cref="TrainModel(string, string, uint)"/> or <see cref="LoadModel(string)"/> first.
        /// </exception>
        public IEnumerable<GameClassificationResult> ClassifyGames(IEnumerable<GameInfo> games)
        {
            if (_predictor == null) throw new InvalidOperationException("You must train or load a model before predicting ESRB ratings");

            foreach (GameInfo game in games)
            {
                ESRBPrediction prediction = _predictor.Predict(game);

                yield return new GameClassificationResult(prediction, game);
            }
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
    }
}