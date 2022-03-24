using MattEland.AI.MLNet.ESRBPredictor.Core;
using Microsoft.AspNetCore.Mvc;

namespace MattEland.AI.MLNet.ESRBPredictor.TraditionalAPI.Controllers;

[ApiController]
[Route("[controller]")]
public class GameRatingController : ControllerBase
{
    private readonly ESRBRatingPredictor _predictor;

    public GameRatingController(ESRBRatingPredictor predictor)
    {
        _predictor = predictor;
    }

    [HttpPost]
    public ActionResult<ESRBPrediction> Evaluate(GameInfo game)
    {
        ESRBPrediction result = _predictor.ClassifyGame(game);

        return Ok(result);
    }
}