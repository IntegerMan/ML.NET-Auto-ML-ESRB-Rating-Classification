using MattEland.AI.MLNet.ESRBPredictor.Core;

namespace MattEland.AI.MLNet.ESRBPredictor.ConsoleApp
{
    public static class SampleGameDataSource
    {
        public static IEnumerable<GameInfo> SampleGames
        {
            get
            {
                yield return new GameInfo()
                {
                    Title = "Teen Side Scroller",
                    CartoonViolence = true,
                    MildLanguage = true,
                    CrudeHumor = true,
                    Violence = true,
                    MildSuggestiveThemes = true
                };
                yield return new GameInfo()
                {
                    Title = "Kinda Sus",
                    MildCartoonViolence = true
                };
                yield return new GameInfo()
                {
                    Title = "The Earthlings are Coming",
                    MildViolence = true,
                    MildFantasyViolence = true,
                };
                yield return new GameInfo()
                {
                    Title = "Shoddy Surgeon Simulator",
                    BloodAndGore = true,
                    DrugReference = true,
                    PartialNudity = true,
                };
                yield return new GameInfo()
                {
                    Title = "Assistant to the Lawn Service Manager 2022",
                    MildLanguage = true,
                    CrudeHumor = true,
                    AlcoholReference = true,
                };
                yield return new GameInfo()
                {
                    Title = "Intense Shoot-o-rama: Why would anyone play this edition",
                    BloodAndGore = true,
                    DrugReference = true,
                    AlcoholReference = true,
                    Nudity = true,
                    StrongLanguage = true,
                    SexualContent = true,
                    SexualThemes = true,
                    MatureHumor = true,
                    IntenseViolence = true,
                    CrudeHumor = true,
                };
            }
        }
    }
}
