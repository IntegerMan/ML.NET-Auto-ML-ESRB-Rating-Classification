using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MattEland.AI.MLNet.ESRBPredictor.Core
{
    public class GameClassificationResult
    {
        public GameClassificationResult(ESRBPrediction prediction, GameRating game)
        {
            Prediction = prediction;
            Game = game;
        }

        public ESRBPrediction Prediction { get; }
        public GameRating Game { get; }
    }
}
