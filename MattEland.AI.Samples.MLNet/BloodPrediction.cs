using Microsoft.ML.Data;

public class BloodPrediction
{
    [ColumnName("PredictedLabel")]
    public bool Blood { get; set; }

    public float Score { get; set; }
}