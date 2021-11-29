using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MattEland.AI.Samples.MLNet
{
    public class BloodInGamesModelTrainer
    {
        private string filePath;

        public BloodInGamesModelTrainer(string filePath)
        {
            if (string.IsNullOrWhiteSpace(filePath))
            {
                throw new ArgumentException($"'{nameof(filePath)}' cannot be null or whitespace.", nameof(filePath));
            }

            this.filePath = filePath;
        }

        public DataViewSchema? Schema { get; private set; }
        public ITransformer? BestModel { get; private set; }

        public ExperimentResult<BinaryClassificationMetrics> Train(MLContext context, uint maxMinutes)
        {
            IDataView trainDataView = context.Data.LoadFromTextFile<GameRating>(filePath, hasHeader: true, separatorChar: ',');

            var experimentSettings = new BinaryExperimentSettings()
            {
                MaxExperimentTimeInSeconds = maxMinutes * 60,
                CacheDirectoryName = null, // Run in memory
            };

            var experiment = context.Auto().CreateBinaryClassificationExperiment(experimentSettings);

            var result = experiment.Execute(trainDataView,
                labelColumnName: "Blood",
                samplingKeyColumn: null,
                progressHandler: new BinaryExperimentProgressHandler());

            this.Schema = trainDataView.Schema;

            this.BestModel = result.BestRun.Model;

            return result;
        }

    }
}
