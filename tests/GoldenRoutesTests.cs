using System.Text.Json;
using Xunit;

namespace RoutingService.Tests;

public class RoutingFeaturesTests
{
    private static float[] Compute(double olat, double olon, double dlat, double dlon)
    {
        var f = new float[RoutingFeatures.Count];
        RoutingFeatures.Compute(olat, olon, dlat, dlon, f);
        return f;
    }

    [Fact]
    public void FeatureCountMatchesModelContract()
    {
        Assert.Equal(8, RoutingFeatures.Count);
        Assert.Equal(
            ["orig_lat", "orig_lon", "dest_lat", "dest_lon", "haversine_dist_m", "bearing_deg", "lat_delta", "lon_delta"],
            RoutingFeatures.Names);
    }

    [Fact]
    public void HaversineMatchesOneDegreeOfLatitude()
    {
        // On a sphere of R = 6371.0088 km, one degree of latitude is pi/180 * R.
        var f = Compute(0, 0, 1, 0);
        Assert.Equal(111_194.93, f[4], 1.0);
    }

    [Fact]
    public void HaversineIsZeroForIdenticalPoints()
    {
        var f = Compute(53.4808, -2.2426, 53.4808, -2.2426);
        Assert.Equal(0f, f[4], 0.001f);
    }

    [Theory]
    [InlineData(0, 0, 0, 1, 90.0)]    // due east
    [InlineData(0, 0, 1, 0, 0.0)]     // due north
    [InlineData(0, 0, 0, -1, 270.0)]  // due west
    [InlineData(0, 0, -1, 0, 180.0)]  // due south
    public void BearingCardinalsAreExact(double olat, double olon, double dlat, double dlon, double expected)
    {
        var f = Compute(olat, olon, dlat, dlon);
        Assert.Equal(expected, f[5], 0.01);
    }

    [Fact]
    public void BearingIsNormalisedToZeroTo360()
    {
        var f = Compute(53.4808, -2.2426, 53.4631, -2.2913);
        Assert.InRange(f[5], 0.0, 360.0);
    }

    [Fact]
    public void DeltasAndRawCoordinatesArePassedThroughUnchanged()
    {
        var f = Compute(53.4808, -2.2426, 53.4631, -2.2913);
        Assert.Equal(53.4808f, f[0], 1e-5f);
        Assert.Equal(-2.2426f, f[1], 1e-5f);
        Assert.Equal(53.4631f, f[2], 1e-5f);
        Assert.Equal(-2.2913f, f[3], 1e-5f);
        Assert.Equal(53.4631f - 53.4808f, f[6], 1e-5f);
        Assert.Equal(-2.2913f - (-2.2426f), f[7], 1e-5f);
    }

    [Fact]
    public void FeaturesMatchPythonReferenceVector()
    {
        // Reference values produced by python/train_export_onnx.py::geometric_features
        // for Manchester Piccadilly -> Manchester Victoria.
        var f = Compute(53.4770, -2.2309, 53.4875, -2.2427);
        Assert.Equal(1_404.567, f[4], 2.0);
        Assert.Equal(326.232, f[5], 1.0);
    }
}

public class GoldenRoutesTests
{
    // The service returns one-decimal JSON values; 0.11 absorbs that rounding.
    private const double ReproducibilityTolerance = 0.11;

    private static readonly Lazy<JsonElement> Fixture = new(LoadFixture);
    private static readonly Lazy<RoutePredictor> Predictor = new(() =>
        new RoutePredictor(Path.Combine(AppContext.BaseDirectory, "models", "model.bin")));

    private static JsonElement LoadFixture()
    {
        var path = Path.Combine(AppContext.BaseDirectory, "golden_routes.json");
        if (!File.Exists(path))
        {
            throw new FileNotFoundException(
                $"Missing {path}. Generate it with: cd python && uv run ../tests/benchmark.py", path);
        }
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement.Clone();
    }

    private static JsonElement Route(string name)
    {
        foreach (var route in Fixture.Value.GetProperty("routes").EnumerateArray())
        {
            if (route.GetProperty("name").GetString() == name) return route;
        }
        throw new Xunit.Sdk.XunitException($"golden route '{name}' not present in fixture");
    }

    public static IEnumerable<object[]> RouteNames()
    {
        foreach (var route in Fixture.Value.GetProperty("routes").EnumerateArray())
        {
            yield return new object[] { route.GetProperty("name").GetString()! };
        }
    }

    private static (double DurationS, double DistanceM) Predict(JsonElement route)
    {
        var orig = route.GetProperty("orig");
        var dest = route.GetProperty("dest");
        return Predictor.Value.Predict(
            orig[0].GetDouble(), orig[1].GetDouble(),
            dest[0].GetDouble(), dest[1].GetDouble());
    }

    [Fact]
    public void FixtureCoversSeveralDistinctRoutes()
    {
        Assert.True(Fixture.Value.GetProperty("routes").GetArrayLength() >= 10);
    }

    /// <summary>
    /// Regression guard: the shipped model must keep producing the values that
    /// were recorded when the fixture was generated.
    /// </summary>
    [Theory]
    [MemberData(nameof(RouteNames))]
    public void ModelReproducesRecordedApproximations(string name)
    {
        var route = Route(name);
        var (duration, distance) = Predict(route);

        Assert.Equal(route.GetProperty("approx_duration_s").GetDouble(), duration, ReproducibilityTolerance);
        Assert.Equal(route.GetProperty("approx_distance_m").GetDouble(), distance, ReproducibilityTolerance);
    }

    /// <summary>Accuracy budget against exact OSRM on the golden set.</summary>
    [Fact]
    public void GoldenRoutesStayWithinAccuracyBudget()
    {
        var durationApe = new List<double>();
        var distanceApe = new List<double>();

        foreach (var route in Fixture.Value.GetProperty("routes").EnumerateArray())
        {
            var (duration, distance) = Predict(route);
            durationApe.Add(Math.Abs(duration - route.GetProperty("osrm_duration_s").GetDouble())
                             / route.GetProperty("osrm_duration_s").GetDouble());
            distanceApe.Add(Math.Abs(distance - route.GetProperty("osrm_distance_m").GetDouble())
                            / route.GetProperty("osrm_distance_m").GetDouble());
        }

        Assert.InRange(Median(durationApe) * 100.0, 0.0, 35.0);
        Assert.InRange(Median(distanceApe) * 100.0, 0.0, 35.0);
    }

    /// <summary>
    /// Regression guard on the headline benchmark numbers recorded over 600
    /// random held-out pairs.
    /// </summary>
    [Fact]
    public void RecordedBenchmarkStaysWithinBudget()
    {
        var accuracy = Fixture.Value.GetProperty("accuracy");
        Assert.InRange(accuracy.GetProperty("duration").GetProperty("medape_pct").GetDouble(), 0.0, 35.0);
        Assert.InRange(accuracy.GetProperty("distance").GetProperty("medape_pct").GetDouble(), 0.0, 35.0);
    }

    [Theory]
    [MemberData(nameof(RouteNames))]
    public void PredictionsAreFiniteAndPositive(string name)
    {
        var (duration, distance) = Predict(Route(name));
        Assert.True(double.IsFinite(duration) && duration > 0, $"duration={duration}");
        Assert.True(double.IsFinite(distance) && distance > 0, $"distance={distance}");
    }

    [Fact]
    public void DistanceGrowsWithSeparation()
    {
        var near = Predictor.Value.Predict(53.4808, -2.2426, 53.4820, -2.2400);
        var far = Predictor.Value.Predict(53.4808, -2.2426, 53.5780, -2.4282);
        Assert.True(far.DistanceM > near.DistanceM * 10, $"{far.DistanceM} vs {near.DistanceM}");
        Assert.True(far.DurationS > near.DurationS * 5);
    }

    private static double Median(List<double> values)
    {
        var sorted = values.OrderBy(v => v).ToArray();
        return sorted.Length % 2 == 1
            ? sorted[sorted.Length / 2]
            : 0.5 * (sorted[sorted.Length / 2 - 1] + sorted[sorted.Length / 2]);
    }
}
