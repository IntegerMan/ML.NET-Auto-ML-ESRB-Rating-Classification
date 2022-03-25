using Microsoft.ML.Data;

namespace MattEland.AI.MLNet.ESRBPredictor.Core;

public class ESRBPrediction
{
    [ColumnName("PredictedLabel")]
    public string ESRBRating { get; set; }

    public float[] Score { get; set; } // [ P:E, P:ET, P:M, P:T]

    public float Confidence => Score.Max();

    public float EveryoneProbability => Score[0];
    public float EveryoneTenPlusProbability => Score[1];
    public float TeenProbability => Score[3];
    public float MatureProbability => Score[2];

    public override string ToString()
    {
        return $"E: {EveryoneProbability:P}, ET: {EveryoneTenPlusProbability:P}, T: {TeenProbability:P}, M: {MatureProbability:P}";
    }

}