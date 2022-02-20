namespace MattEland.AI.MLNet.ESRBPredictor.Core
{
    public class GameClassificationResult
    {
        private readonly ESRBPrediction _prediction;

        public GameClassificationResult(ESRBPrediction prediction, GameInfo game)
        {
            _prediction = prediction;
            Game = game.Title;
        }

        public string Game { get; }

        public string Prediction => _prediction.ESRBRating;
        public float Confidence => _prediction.Confidence;

        public float EveryoneProbability => _prediction.Score[0];
        public float EveryoneTenPlusProbability => _prediction.Score[1];
        public float TeenProbability => _prediction.Score[3];
        public float MatureProbability => _prediction.Score[2];

        public override string ToString()
        {
            return $"E: {EveryoneProbability:P}, ET: {EveryoneTenPlusProbability:P}, T: {TeenProbability:P}, M: {MatureProbability:P}";
        }
    }
}
