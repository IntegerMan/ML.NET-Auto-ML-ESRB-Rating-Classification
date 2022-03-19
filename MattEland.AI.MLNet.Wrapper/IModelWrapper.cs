using Microsoft.ML.AutoML;

namespace MattEland.AI.MLNet.Wrapper;

public interface IModelWrapper<TMetrics, in TModel, out TPredictionResult>
{
    /// <summary>
    /// Trains a machine learning model based on training and validation data in the
    /// <paramref name="trainFilePath"/> and <paramref name="validationFilePath"/>.
    /// Once a model is trained, the <see cref=Predict>Predict</see> method can be called to
    /// predict values, or the <see cref="SaveModel">SaveModel</see> method can be called to
    /// save the model for future runs.
    /// </summary>
    /// <param name="trainFilePath">The file containing the comma separated values for model training</param>
    /// <param name="validationFilePath">The file containing the comma separated values for model validation</param>
    /// <param name="secondsToTrain">The number of seconds to spend training the machine learning model</param>
    /// <returns>Information about the best model</returns>
    RunDetail<TMetrics> TrainModel(
        string trainFilePath, 
        string validationFilePath, 
        uint secondsToTrain);

    /// <summary>
    /// Predicts the label column of a specified <paramref name="model"/> and returns it
    /// </summary>
    /// <param name="model">The model to predict the label of</param>
    /// <returns>An ESRB prediction including certainty factors</returns>
    /// <exception cref="InvalidOperationException">
    /// Thrown if no model has been trained. Call <see cref="TrainModel"/> or <see cref="LoadModel"/> first.
    /// </exception>
    TPredictionResult Predict(TModel model);

    /// <summary>
    /// Saves the model to disk
    /// </summary>
    /// <param name="filename">The path of the model file to save</param>
    /// <exception cref="InvalidOperationException">
    /// Thrown if no model has been trained. Call <see cref="TrainModel"/> or <see cref="LoadModel"/> first.
    /// </exception>
    void SaveModel(string filename);

    /// <summary>
    /// Loads the model from disk
    /// </summary>
    /// <param name="filename">The path of the model file to load</param>
    void LoadModel(string filename);
}