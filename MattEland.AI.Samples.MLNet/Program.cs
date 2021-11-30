using MattEland.AI.Samples.MLNet;

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
            string? input = Console.ReadLine();
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
                        IEnumerable<GameRating> games = SampleGameDataSource.SampleGames;
                        predictor.ClassifyGames(games);
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
        Console.WriteLine("How many minutes do you want to train?");
        string? minutesStr = Console.ReadLine();
        Console.WriteLine();

        if (!int.TryParse(minutesStr, out int minutesToTrain))
        {
            Console.WriteLine("Invalid minute input. Expecting a positive whole number.");
            return;
        }

        if (minutesToTrain <= 0)
        {
            Console.WriteLine("You must train for at least one minute");
            return;
        }

        string minutesText = minutesToTrain == 1 ? "1 minute" : minutesToTrain + " minutes";

        Console.WriteLine($"Training model now.... This will take around {minutesText}");
        Console.WriteLine();

        string confusionMatrix = predictor.TrainModel("ESRB.csv", (uint)minutesToTrain);

        Console.WriteLine();
        Console.WriteLine("Training completed!");
        Console.WriteLine();
        Console.WriteLine(confusionMatrix);
    }
}