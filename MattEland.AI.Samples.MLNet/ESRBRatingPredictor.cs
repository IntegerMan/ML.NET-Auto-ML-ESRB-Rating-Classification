using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using System.Text;

namespace MattEland.AI.MLNet.ESRBPredictor 
{

    /// <summary>
    /// Manages machine learning models for classifying video games into an ESRB rating given a set of traits about the game's content.
    /// </summary>
    public class ESRBRatingPredictor
    {
        private readonly MLContext _context = new MLContext();

        private ITransformer? _model;
        private DataViewSchema? _schema;

        /// <summary>
        /// <para>
        /// Trains a machine learning model based on ESRB game data in the <paramref name="trainFilePath"/> and <paramref name="validationFilePath"/>.
        /// Once a model is trained, the <see cref="ClassifyGames(IEnumerable{GameRating})">ClassifyGames</see> method can be called to
        /// predict ESRB ratings, or the <see cref="SaveModel(string)">SaveModel</see> method can be called to save the model for future runs.
        /// </para>
        /// <para>
        /// See <see href="https://www.kaggle.com/imohtn/video-games-rating-by-esrb">Kaggle</see> for the dataset used for this application.
        /// </para>
        /// <para>
        /// This method will return the name of the best model and a confusion matrix indicating the performance of that model.
        /// </para>
        /// </summary>
        /// <param name="trainFilePath">The file containing the comma separated values for model training</param>
        /// <param name="validationFilePath">The file containing the comma separated values for model validation</param>
        /// <param name="secondsToTrain">The number of seconds to spend training the machine learning model</param>
        /// <returns>Information about the best model and a Confusion Matrix representing that model's performance</returns>
        public string TrainModel(string trainFilePath, string validationFilePath, uint secondsToTrain)
        {
            // Load data. This was built around the Kaggle dataset at https://www.kaggle.com/imohtn/video-games-rating-by-esrb
            IDataView trainData = _context.Data.LoadFromTextFile<GameRating>(trainFilePath, separatorChar: ',', hasHeader: true, allowQuoting: true);
            IDataView validationData = _context.Data.LoadFromTextFile<GameRating>(validationFilePath, separatorChar: ',', hasHeader: true, allowQuoting: true);

            // Configure the experiment
            MulticlassExperimentSettings settings = new MulticlassExperimentSettings()
            {
                OptimizingMetric = MulticlassClassificationMetric.LogLoss,
                MaxExperimentTimeInSeconds = secondsToTrain,
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

            // Return a formatted matrix tor analyzing model performance
            StringBuilder sb = new StringBuilder();
            sb.AppendLine($"Best algorithm: {result.BestRun.TrainerName}");
            sb.AppendLine(result.BestRun.ValidationMetrics.ConfusionMatrix.GetFormattedConfusionTable());

            return sb.ToString();
        }

        /// <summary>
        /// Predicts the ESRB ratings of incoming <paramref name="games"/> and returns those predictions
        /// </summary>
        /// <param name="games">The games to classify</param>
        /// <returns>A series of ESRB predictions including certainty factors</returns>
        /// <exception cref="InvalidOperationException">
        /// Thrown if no model has been trained. Call <see cref="TrainModel(string, string, uint)"/> or <see cref="LoadModel(string)"/> first.
        /// </exception>
        public IEnumerable<Tuple<ESRBPrediction, GameRating>> ClassifyGames(IEnumerable<GameRating> games)
        {
            if (_model == null) throw new InvalidOperationException("You must train or load a model before predicting ESRB ratings");

            PredictionEngine<GameRating, ESRBPrediction> predictEngine =
                _context.Model.CreatePredictionEngine<GameRating, ESRBPrediction>(transformer: _model, inputSchema: _schema);

            foreach (GameRating game in games)
            {
                ESRBPrediction prediction = predictEngine.Predict(game);

                yield return new (prediction, game);
            }
        }

        /// <summary>
        /// Loads the model from disk
        /// </summary>
        /// <param name="filename">The path of the model file to load</param>
        public void LoadModel(string filename)
        {
            _model = _context.Model.Load(filename, out _schema);
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