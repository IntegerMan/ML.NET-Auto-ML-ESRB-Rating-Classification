using MattEland.AI.MLNet.ESRBPredictor.Core;
using Microsoft.ML;
using Microsoft.ML.AutoML;

namespace MattEland.AI.MLNet.Wrapper;

public abstract class ModelWrapperBase<TMetrics, TModel, TPrediction> : IModelWrapper<TMetrics, TModel, TPrediction> where TModel : class where TPrediction : class, new()
{
    private ITransformer? _model;
    private PredictionEngine<TModel, TPrediction>? _predictor;
    private DataViewSchema? _schema;

    protected MLContext Context { get; } = new();

    public abstract RunDetail<TMetrics> TrainModel(
        string trainFilePath, 
        string validationFilePath, 
        uint secondsToTrain);

    public abstract TPrediction Predict(TModel model);

    protected PredictionEngine<TModel, TPrediction> BuildPredictionEngine() =>
        Context.Model.CreatePredictionEngine<TModel, TPrediction>(
            transformer: _model,
            inputSchema: _schema);

    protected PredictionEngine<TModel, TPrediction> EnsurePredictorExists()
    {
        if (_model == null)
        {
            throw new InvalidOperationException("You must train or load a model before predicting");
        }

        _predictor ??= BuildPredictionEngine();

        return _predictor;
    }

    protected void UpdateModel(ITransformer model)
    {
        _model = model;
        _predictor = null;
    }


    protected void UpdateSchema(DataViewSchema schema)
    {
        _schema = schema;
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
        if (_model == null)
        {
            throw new InvalidOperationException("You must train or load a model before saving");
        }

        Context.Model.Save(_model, _schema, filename);
    }



    /// <summary>
    /// Loads the model from disk
    /// </summary>
    /// <param name="filename">The path of the model file to load</param>
    public void LoadModel(string filename)
    {
        _model = Context.Model.Load(filename, out _schema);

        // Whenever our model changes, it's nice to update the prediction engine
        _predictor = BuildPredictionEngine();
    }
}