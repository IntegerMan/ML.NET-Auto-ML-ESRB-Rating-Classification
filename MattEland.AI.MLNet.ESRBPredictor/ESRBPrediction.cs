using Microsoft.ML.Data;

namespace MattEland.AI.MLNet.ESRBPredictor
{
    public class ESRBPrediction
    {
        [ColumnName("PredictedLabel")]
        public string ESRBRating { get; set; }

        public float[] Score { get; set; }

        public float Confidence => Score.Max();
    }
}