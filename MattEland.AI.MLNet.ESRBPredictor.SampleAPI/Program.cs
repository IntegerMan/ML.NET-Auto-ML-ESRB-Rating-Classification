using MattEland.AI.MLNet.ESRBPredictor.Core;
using Microsoft.AspNetCore.Mvc;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
// Learn more about configuring Swagger/OpenAPI at https://aka.ms/aspnetcore/swashbuckle
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

var app = builder.Build();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseHttpsRedirection();


// App Code Starts Here
ESRBRatingPredictor predictor = new ESRBRatingPredictor();
predictor.LoadModel("Model.zip");

app.MapPost("/esrb-predictor", (GameRating game) =>
    {
        GameClassificationResult prediction = predictor.ClassifyGame(game);

        return Results.Ok(prediction);
    })
.WithName("PredictESRBRating");


// Start the Web Application
app.Run();