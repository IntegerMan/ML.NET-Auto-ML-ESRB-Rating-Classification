using Microsoft.ML.Data;

namespace MattEland.AI.MLNet.ESRBPredictor.Core
{
    public class ESRBPrediction
    {
        [ColumnName("PredictedLabel")]
        public string ESRBRating { get; set; }

        [ColumnName("Score")]
        public float[] Score { get; set; }
    }
}