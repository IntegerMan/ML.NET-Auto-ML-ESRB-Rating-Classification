using MattEland.AI.MLNet.ESRBPredictor.Core;
using Microsoft.AspNetCore.Mvc;

namespace MattEland.AI.MLNet.ESRBPredictor.TraditionalAPI.Controllers
{
    [ApiController]
    [Route("[controller]")]
    public class GameRatingController : ControllerBase
    {
        private readonly Core.ESRBPredictor _predictor;

        public GameRatingController(Core.ESRBPredictor predictor)
        {
            _predictor = predictor;
        }

        [HttpPost]
        public ActionResult<ESRBPrediction> Evaluate(GameInfo game)
        {
            ESRBPrediction result = _predictor.Predict(game);

            return Ok(result);
        }
    }
}