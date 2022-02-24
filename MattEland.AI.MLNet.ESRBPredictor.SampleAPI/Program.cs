using MattEland.AI.MLNet.ESRBPredictor.Core;

#region Auto-Generated Code
// ASP .NET Core 6 Minimal API Boilerplate
var builder = WebApplication.CreateBuilder(args);

builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

var app = builder.Build();
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}
app.UseHttpsRedirection();
#endregion

// Application Startup Code
ESRBRatingPredictor predictor = new ESRBRatingPredictor();
predictor.LoadModel("Model.zip");

// Endpoint Mapping
app.MapPost("/esrb-predictor", (GameInfo game) => Results.Ok(predictor.ClassifyGame(game)));

// Start the Web Application
app.Run();
