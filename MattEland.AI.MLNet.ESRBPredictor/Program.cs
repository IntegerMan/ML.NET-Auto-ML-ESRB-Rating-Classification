using MattEland.AI.MLNet.ESRBPredictor.Core;

namespace MattEland.AI.MLNet.ESRBPredictor.ConsoleApp
{
    public static class Program
    {
        public static void Main()
        {
            ESRBRatingPredictor predictor = new ESRBRatingPredictor();
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
                    switch (input?.ToUpperInvariant())
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
                            IEnumerable<GameInfo> games = SampleGameDataSource.SampleGames;
                            foreach (GameClassificationResult result in predictor.ClassifyGames(games))
                            {
                                string title = result.Game.Title;
                                string rating = result.Prediction.ESRBRating;
                                float confidence = result.Prediction.Confidence;

                                Console.WriteLine($"Predicting rating of {rating} for \"{title}\" with a confidence score of {confidence:p}");
                            }
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

        private static void HandleSaveModel(ESRBRatingPredictor predictor, string modelFile)
        {
            try
            {
                predictor.SaveModel(modelFile);
                Console.WriteLine("Model saved.");
            }
            catch (IOException ex)
            {
                Console.WriteLine($"Could not save the model to {modelFile}: {ex.Message}");
            }
        }

        private static void HandleLoadModel(ESRBRatingPredictor predictor, string modelFile)
        {
            try
            {
                predictor.LoadModel(modelFile);
                Console.WriteLine("Model loaded.");
            }
            catch (IOException ex)
            {
                Console.WriteLine($"Could not load the model from {modelFile}: {ex.Message}");
            }
        }

        private static void HandleTrainModel(ESRBRatingPredictor predictor)
        {
            Console.WriteLine("How many seconds do you want to train? (10 Recommended)");
            string? minutesStr = Console.ReadLine();
            Console.WriteLine();

            if (!int.TryParse(minutesStr, out int secondsToTrain))
            {
                Console.WriteLine("Invalid minute input. Expecting a positive whole number.");
                return;
            }

            if (secondsToTrain <= 0)
            {
                Console.WriteLine("You must train for at least one second");
                return;
            }

            string timeText = secondsToTrain == 1 ? "1 second" : secondsToTrain + " seconds";

            Console.WriteLine($"Training model now.... This will take around {timeText}");
            Console.WriteLine();

            string confusionMatrix = predictor.TrainModel("ESRB.csv", "ESRBTest.csv", (uint)secondsToTrain);

            Console.WriteLine();
            Console.WriteLine("Training completed!");
            Console.WriteLine();
            Console.WriteLine(confusionMatrix);
        }
    }
}