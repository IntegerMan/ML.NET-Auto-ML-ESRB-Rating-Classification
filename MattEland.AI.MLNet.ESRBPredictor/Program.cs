using System.Text;
using MattEland.AI.MLNet.ESRBPredictor.Core;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;

namespace MattEland.AI.MLNet.ESRBPredictor.ConsoleApp
{
    public static class Program
    {
        public static void Main()
        {
            Core.ESRBPredictor predictor = new();
            string modelFile = Path.Combine(Environment.CurrentDirectory, "Model.zip");

            bool shouldQuit = false;

            Console.WriteLine("Welcome to the ESRB Predictor by Matt Eland (@IntegerMan)");

            do
            {
                Console.WriteLine();
                Console.WriteLine("What would you like to do?");
                Console.WriteLine();
                Console.WriteLine("(T)rain a model");
                Console.WriteLine("(S)ave the model to disk");
                Console.WriteLine("(L)oad the last saved model from disk");
                Console.WriteLine("(P)redict ESRB ratings");
                Console.WriteLine("(Q)uit");
                Console.WriteLine();

                Console.Write("> ");
                string input = Console.ReadLine()!;
                Console.WriteLine();

                try
                {
                    switch (input.ToUpperInvariant())
                    {
                        case "T": // Train a model
                            HandleTrainModel(predictor);
                            break;

                        case "S": // Save a model
                            HandleSaveModel(predictor, modelFile);
                            break;

                        case "L": // Load a saved model
                            HandleLoadModel(predictor, modelFile);
                            break;

                        case "P": // Predict ESRB ratings
                            PredictGameRatings(predictor);
                            break;

                        case "Q": // Quit
                            shouldQuit = true;
                            Console.WriteLine("Thanks for using the classifier!");
                            break;

                        default:
                            Console.WriteLine("Invalid input. Please type T, S, L, P, or Q");
                            break;
                    }
                }
                catch (InvalidOperationException ex)
                {
                    Console.WriteLine(ex.Message); // These are built in such a way that they should be human-readable
                }
            } while (!shouldQuit);

        }

        private static void PredictGameRatings(Core.ESRBPredictor predictor)
        {
            List<GameInfo> games = SampleGameDataSource.SampleGames.ToList();

            // TODO: Add any custom games for demo purposes here

            foreach (GameInfo game in games)
            {
                ESRBPrediction result = predictor.Predict(game);

                Console.WriteLine($"Predicting rating of {result.ESRBRating} for \"{game.Title}\"");
                Console.WriteLine($"\tDetails: {result}");
                Console.WriteLine();
            }
        }

        private static void HandleTrainModel(Core.ESRBPredictor predictor)
        {
            uint secondsToTrain = PromptForSecondsToTrain();

            Console.WriteLine($"Training model now.... This will take around {secondsToTrain} second(s)");
            Console.WriteLine();

            RunDetail<MulticlassClassificationMetrics> bestRun = 
                predictor.TrainModel("ESRB.csv", "ESRBTest.csv", secondsToTrain);

            Console.WriteLine();
            Console.WriteLine("Training completed!");
            Console.WriteLine();

            Console.WriteLine($"Best algorithm: {bestRun.TrainerName}");
            Console.WriteLine(bestRun.ValidationMetrics.ConfusionMatrix.GetFormattedConfusionTable());
        }

        private static void HandleSaveModel(Core.ESRBPredictor predictor, string modelFile)
        {
            try
            {
                predictor.SaveModel(modelFile);

                Console.WriteLine($"Model saved to {modelFile}");
            }
            catch (IOException ex)
            {
                Console.WriteLine($"Could not save the model to {modelFile}: {ex.Message}");
            }
        }

        private static void HandleLoadModel(Core.ESRBPredictor predictor, string modelFile)
        {
            try
            {
                predictor.LoadModel(modelFile);

                Console.WriteLine($"Model loaded from {modelFile}");
            }
            catch (IOException ex)
            {
                Console.WriteLine($"Could not load the model from {modelFile}: {ex.Message}");
            }
        }

        private static uint PromptForSecondsToTrain()
        {
            // Loop until we get a valid input
            while (true)
            {
                Console.WriteLine("How many seconds do you want to train? (10 Recommended)");
                string minutesStr = Console.ReadLine()!;
                Console.WriteLine();

                // Default to 10 if the user just hits enter
                if (string.IsNullOrWhiteSpace(minutesStr))
                {
                    minutesStr = "10";
                    Console.WriteLine(minutesStr);
                }

                // Ensure the input is valid
                if (int.TryParse(minutesStr, out int secondsToTrain))
                {
                    if (secondsToTrain > 0)
                    {
                        return (uint)secondsToTrain;
                    }

                    Console.WriteLine("You must train for at least one second");
                }
                else
                {
                    Console.WriteLine("Invalid minute input. Expecting a positive whole number.");
                }
            }
        }

    }
}