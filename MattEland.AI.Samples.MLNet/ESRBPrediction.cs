using Microsoft.ML.Data;

public class ESRBPrediction
{
    [ColumnName("PredictedLabel")]
    public string ESRBRating { get; set; }

    public float[] Score { get; set; }
}