using MattEland.AI.MLNet.ESRBPredictor.Core;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.

builder.Services.AddControllers();
// Learn more about configuring Swagger/OpenAPI at https://aka.ms/aspnetcore/swashbuckle
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

// Create the globally-used ESRB Predictor and load the trained model into it
ESRBRatingPredictor predictor = new();
predictor.LoadModel("Model.zip");

// Whenever a controller needs an ESRBRatingPredictor, give it this instance
builder.Services.AddSingleton<ESRBRatingPredictor>(predictor);


var app = builder.Build();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseHttpsRedirection();

app.UseAuthorization();

app.MapControllers();

app.Run();
