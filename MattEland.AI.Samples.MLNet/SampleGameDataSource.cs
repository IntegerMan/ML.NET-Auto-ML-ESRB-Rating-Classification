using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MattEland.AI.Samples.MLNet
{
    public static class SampleGameDataSource
    {
        public static IEnumerable<GameRating> SampleGames
        {
            get
            {
                yield return new GameRating()
                {
                    Title = "Teen Side Scroller",
                    AlcoholReference = true,
                    CartoonViolence = true,
                    MildLanguage = true,
                    CrudeHumor = true,
                    Violence = true,
                    MildSuggestiveThemes = true,
                    ESRBRating = "T"
                };
                yield return new GameRating()
                {
                    Title = "Shoddy Surgeon Simulator",
                    BloodAndGore = true,
                    DrugReference = true,
                    PartialNudity = true,
                    ESRBRating = "M"
                };
                yield return new GameRating()
                {
                    Title = "Assistant to the Lawn Service Manager 2022",
                    CrudeHumor = true,
                    MildLanguage = true,
                    ESRBRating = "E"
                };
                yield return new GameRating()
                {
                    Title = "Intense Shoot-o-rama: Why would anyone play this edition",
                    BloodAndGore = true,
                    DrugReference = true,
                    PartialNudity = true,
                    StrongLanguage = true,
                    SexualContent = true,
                    SexualThemes = true,
                    MatureHumor = true,
                    Lyrics = true,
                    IntenseViolence = true,
                    CrudeHumor = true,
                    ESRBRating = "M"
                };
            }
        }
    }
}
