using MattEland.AI.MLNet.Wrapper;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;

namespace MattEland.AI.MLNet.ESRBPredictor.Core 
{
    /// <summary>
    /// Manages machine learning models for classifying video games into an ESRB rating
    /// given a set of traits about the game's content.
    /// </summary>
    public class ESRBPredictor : ModelWrapperBase<MulticlassClassificationMetrics, GameInfo, ESRBPrediction>
    {





        /// <summary>
        /// Trains a machine learning model based on ESRB game data in the <paramref name="trainFilePath"/> and <paramref name="validationFilePath"/>.
        /// Once a model is trained, the <see cref="Predict">Predict</see> method can be called to
        /// predict ESRB ratings, or the <see cref="SaveModel(string)">SaveModel</see> method can be called to save the model for future runs.
        ///
        /// See <see href="https://www.kaggle.com/imohtn/video-games-rating-by-esrb">Kaggle</see> for the dataset used for this application.
        /// </summary>
        /// <param name="trainFilePath">The file containing the comma separated values for model training</param>
        /// <param name="validationFilePath">The file containing the comma separated values for model validation</param>
        /// <param name="secondsToTrain">The number of seconds to spend training the machine learning model</param>
        /// <returns>Information about the best model</returns>
        public override RunDetail<MulticlassClassificationMetrics> TrainModel(
            string trainFilePath, 
            string validationFilePath, 
            uint secondsToTrain)
        {
            // Load training and test data.
            // See Kaggle dataset at https://www.kaggle.com/imohtn/video-games-rating-by-esrb
            IDataView trainData = Context.Data.LoadFromTextFile<GameInfo>(
                path: trainFilePath,
                separatorChar: ',',
                hasHeader: true,
                allowQuoting: true);

            IDataView validationData = Context.Data.LoadFromTextFile<GameInfo>(
                path: validationFilePath, 
                separatorChar: ',', 
                hasHeader: true, 
                allowQuoting: true);

            // Store the schema to optimize future tasks
            UpdateSchema(trainData.Schema);




            // NOTE: If you don't have two separate files you can split an IDataView
            // using _context.Data.TrainTestSplit() and specify split parameters





            // Configure the experiment
            MulticlassExperimentSettings settings = new()
            {
                OptimizingMetric = MulticlassClassificationMetric.MacroAccuracy,
                MaxExperimentTimeInSeconds = secondsToTrain,
            };

            MulticlassClassificationExperiment experiment = 
                Context.Auto().CreateMulticlassClassificationExperiment(settings);







            // Synchronously Train the model
            ExperimentResult<MulticlassClassificationMetrics> result = 
                experiment.Execute(
                    trainData: trainData,
                    validationData: validationData,
                    labelColumnName: nameof(GameInfo.ESRBRating),
                    progressHandler: new MulticlassConsoleProgressReporter());




            // Process our finished result
            UpdateModel(result.BestRun.Model);

            // Note: We have result.BestRun.ValidationMetrics available to us for evaluation

            // Return the best-performing model. This will include performance metrics
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
        public override ESRBPrediction Predict(GameInfo game)
        {
            PredictionEngine<GameInfo, ESRBPrediction> predictor = EnsurePredictorExists();

            ESRBPrediction prediction = predictor.Predict(game);

            return prediction;
        }


    }
}