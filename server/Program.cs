using System.Diagnostics;
using System.Globalization;
using System.Text.Json.Serialization;

var builder = WebApplication.CreateBuilder(args);

// Predictable default; ASPNETCORE_URLS still wins if the operator sets it.
builder.WebHost.UseUrls(Environment.GetEnvironmentVariable("ASPNETCORE_URLS") ?? "http://localhost:5080");

// A single shared model serves every request; Predict is read-only and
// thread-safe, so no locking or pooling is needed at this scale.
builder.Services.AddSingleton(_ =>
{
    var modelPath = Environment.GetEnvironmentVariable("MODEL_PATH")
                    ?? Path.Combine(AppContext.BaseDirectory, "models", "model.bin");
    if (!File.Exists(modelPath))
    {
        throw new FileNotFoundException(
            $"Tree ensemble model not found at '{modelPath}'. Run `npm run train` " +
            "and `uv run python/export_binary.py`, or set MODEL_PATH.", modelPath);
    }
    return new RoutePredictor(modelPath);
});

var app = builder.Build();

// Reports server-side processing time so latency can be measured without the
// HTTP client's own overhead dominating the numbers.
app.Use(async (context, next) =>
{
    long startedAt = Stopwatch.GetTimestamp();
    context.Response.OnStarting(() =>
    {
        // Server-Timing's dur is milliseconds by specification.
        double millis = Stopwatch.GetElapsedTime(startedAt).TotalMilliseconds;
        context.Response.Headers["Server-Timing"] = string.Create(
            CultureInfo.InvariantCulture, $"app;dur={millis:F3}");
        return Task.CompletedTask;
    });
    await next();
});

app.MapGet("/health", (RoutePredictor p) => Results.Ok(new
{
    status = "ok",
    model = Path.GetFileName(p.ModelPath),
    features = RoutingFeatures.Names,
}));

app.MapGet("/route", (RoutePredictor predictor, string? orig, string? dest) =>
{
    if (!TryParseLatLon(orig, out var oLat, out var oLon))
    {
        return Results.BadRequest(new { error = "orig must be 'lat,lon' with -90<=lat<=90 and -180<=lon<=180" });
    }
    if (!TryParseLatLon(dest, out var dLat, out var dLon))
    {
        return Results.BadRequest(new { error = "dest must be 'lat,lon' with -90<=lat<=90 and -180<=lon<=180" });
    }

    var (durationS, distanceM) = predictor.Predict(oLat, oLon, dLat, dLon);
    return Results.Ok(new RouteResponse(
        DurationS: Math.Round(durationS, 1),
        DistanceM: Math.Round(distanceM, 1)));
});

app.Run();

static bool TryParseLatLon(string? value, out double lat, out double lon)
{
    lat = lon = 0;
    if (string.IsNullOrWhiteSpace(value)) return false;
    var parts = value.Split(',', 2, StringSplitOptions.TrimEntries);
    if (parts.Length != 2) return false;
    if (!double.TryParse(parts[0], NumberStyles.Float, CultureInfo.InvariantCulture, out lat)) return false;
    if (!double.TryParse(parts[1], NumberStyles.Float, CultureInfo.InvariantCulture, out lon)) return false;
    return lat is >= -90 and <= 90 && lon is >= -180 and <= 180;
}

internal sealed record RouteResponse(
    [property: JsonPropertyName("duration_s")] double DurationS,
    [property: JsonPropertyName("distance_m")] double DistanceM);

/// <summary>Feature contract shared with python/train_export_onnx.py -- order matters.</summary>
internal static class RoutingFeatures
{
    public const int Count = 8;
    public const double EarthRadiusM = 6_371_008.8;

    public static readonly string[] Names =
    [
        "orig_lat", "orig_lon", "dest_lat", "dest_lon",
        "haversine_dist_m", "bearing_deg", "lat_delta", "lon_delta",
    ];

    /// <summary>
    /// Writes the eight model inputs for one ordered pair into <paramref name="dst"/>.
    /// Kept allocation-free and branch-light: this runs on every query.
    /// </summary>
    public static void Compute(double origLat, double origLon, double destLat, double destLon, Span<float> dst)
    {
        const double deg2rad = Math.PI / 180.0;

        double oLatR = origLat * deg2rad;
        double oLonR = origLon * deg2rad;
        double dLatR = destLat * deg2rad;
        double dLonR = destLon * deg2rad;

        double dLat = dLatR - oLatR;
        double dLon = dLonR - oLonR;

        double sinHalfLat = Math.Sin(dLat * 0.5);
        double sinHalfLon = Math.Sin(dLon * 0.5);
        double a = sinHalfLat * sinHalfLat + Math.Cos(oLatR) * Math.Cos(dLatR) * sinHalfLon * sinHalfLon;
        if (a < 0.0) a = 0.0; else if (a > 1.0) a = 1.0;
        double haversine = 2.0 * EarthRadiusM * Math.Asin(Math.Sqrt(a));

        double y = Math.Sin(dLon) * Math.Cos(dLatR);
        double x = Math.Cos(oLatR) * Math.Sin(dLatR) - Math.Sin(oLatR) * Math.Cos(dLatR) * Math.Cos(dLon);
        double bearing = Math.Atan2(y, x) * (180.0 / Math.PI);
        if (bearing < 0.0) bearing += 360.0;

        dst[0] = (float)origLat;
        dst[1] = (float)origLon;
        dst[2] = (float)destLat;
        dst[3] = (float)destLon;
        dst[4] = (float)haversine;
        dst[5] = (float)bearing;
        dst[6] = (float)(destLat - origLat);
        dst[7] = (float)(destLon - origLon);
    }
}

internal sealed class RoutePredictor
{
    private readonly TreeEnsembleModel _model;
    private readonly float[] _features = new float[RoutingFeatures.Count];
    private readonly float[] _outputs = new float[2];

    public RoutePredictor(string modelPath)
    {
        ModelPath = modelPath;
        _model = TreeEnsembleModel.Load(modelPath);
    }

    public string ModelPath { get; }

    public (double DurationS, double DistanceM) Predict(double origLat, double origLon, double destLat, double destLon)
    {
        RoutingFeatures.Compute(origLat, origLon, destLat, destLon, _features);
        return Predict(_features);
    }

    public (double DurationS, double DistanceM) Predict(float[] features)
    {
        // model.bin holds targets in graph order: index 0 = distance_m, 1 = duration_s.
        _model.Predict(features, _outputs);
        return (_outputs[1], _outputs[0]);
    }
}
