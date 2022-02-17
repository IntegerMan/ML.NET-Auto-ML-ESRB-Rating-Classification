using Microsoft.ML.Data;

namespace MattEland.AI.MLNet.ESRBPredictor.Core
{
    public class ESRBPrediction
    {
        [ColumnName("PredictedLabel")]
        public string ESRBRating { get; set; }

        public float[] Score { get; set; }

        public float Confidence => Score.Max();
    }
}