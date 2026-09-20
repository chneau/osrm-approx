using System.Buffers.Binary;
using System.Runtime.InteropServices;

/// <summary>
/// E11 (see IMPROVEMENTS.md): a tiny, dependency-free interpreter for the
/// LightGBM tree ensemble, loaded from <c>model.bin</c> as emitted by
/// <c>python/export_binary.py</c>. Replaces ONNX Runtime, which was the dominant
/// RSS cost; the trees and leaf values are identical to the ONNX graph.
/// </summary>
/// <remarks>
/// Binary layout (little-endian), written by <c>python/export_binary.py</c>:
/// <code>
///   magic       char[4]  "OSRT"
///   version     uint32   2
///   flags       uint32   bit0 = leaf values are uint16-quantised
///   n_features  uint32
///   n_targets   uint32
///   per target (graph order: distance_m, duration_s):
///     base_value    float32
///     value_scale   float32           decode: (q - 32768) * value_scale
///     n_trees       uint32
///     n_nodes       uint32
///     tree_offsets  int32[n_trees+1]  start row of each tree's node block
///     feature       uint8[n_nodes]
///     threshold     float32[n_nodes]
///     left          int32[n_nodes]    internal: row of the x&lt;=threshold child;
///                                     leaf: ~row (negative)
///     right         int32[n_nodes]    internal: row of the x&gt;threshold child
///     value         float32[n_nodes]  (flags bit0 clear)
///     value_q       uint16[n_nodes]   (flags bit0 set)
/// </code>
/// Child ids are already rebased to global rows, so the walk never needs the
/// per-tree offset once the (feature, threshold, left, right) row is chosen.
/// Internal nodes use the BRANCH_LEQ mode: <c>x &lt;= threshold</c> takes the left
/// child (matching ONNX Runtime's TreeEnsembleRegressor semantics).
/// </remarks>
/// <remarks>
/// Two deliberate width choices keep both the file and the in-memory tables narrow,
/// because after the streaming load it is the resident arrays, not the file, that RSS
/// tracks: features are indices into an 8-wide vector so they are <c>byte</c>, and a
/// leaf is marked by a negative <c>left</c> instead of a stored flag, so no per-node
/// flag array exists. <c>Load</c> also streams the file rather than reading it whole,
/// so no second, full-size copy of the model is ever held.
/// </remarks>
internal sealed class TreeEnsembleModel
{
    private const int U16Offset = 32768;
    private static readonly byte[] Magic = "OSRT"u8.ToArray();

    // Arrays are indexed [target]. Every query walks all targets, so keeping the
    // per-target tables in flat arrays avoids a double indirection per node.
    // Exactly one of _value / _valueQ is populated, per the file's flags.
    private readonly int[] _nTrees;
    private readonly int[][] _offsets;
    private readonly byte[][] _feature;
    private readonly float[][] _threshold;
    private readonly int[][] _left;
    private readonly int[][] _right;
    private readonly float[][] _value;
    private readonly ushort[][] _valueQ;
    private readonly float[] _valueScale;
    private readonly float[] _baseValue;

    public int FeatureCount { get; }
    public int TargetCount => _nTrees.Length;

    private TreeEnsembleModel(
        int featureCount,
        int[] nTrees,
        int[][] offsets,
        byte[][] feature,
        float[][] threshold,
        int[][] left,
        int[][] right,
        float[][] value,
        ushort[][] valueQ,
        float[] valueScale,
        float[] baseValue)
    {
        FeatureCount = featureCount;
        _nTrees = nTrees;
        _offsets = offsets;
        _feature = feature;
        _threshold = threshold;
        _left = left;
        _right = right;
        _value = value;
        _valueQ = valueQ;
        _valueScale = valueScale;
        _baseValue = baseValue;
    }

    public static TreeEnsembleModel Load(string path)
    {
        // Stream the file instead of File.ReadAllBytes. Reading it whole allocated a
        // byte[] the size of the entire table, on top of the typed node arrays it was
        // parsed into; being large, that buffer went to the Large Object Heap and
        // stayed resident until a gen2 collection, so the model cost ~1.9x its file
        // size in RSS. Each array is now read straight into its final allocation.
        using var stream = new FileStream(
            path, FileMode.Open, FileAccess.Read, FileShare.Read,
            bufferSize: 1 << 16, FileOptions.SequentialScan);

        Span<byte> magic = stackalloc byte[4];
        ReadExactly(stream, magic, path);
        if (!magic.SequenceEqual(Magic))
        {
            throw new InvalidDataException($"'{path}' is not an OSRT tree ensemble (bad magic).");
        }

        uint version = ReadUInt32(stream, path);
        if (version != 2)
        {
            throw new InvalidDataException(
                $"Unsupported model.bin version {version} (expected 2). " +
                "Re-run `uv run python/export_binary.py` to recompile the model.");
        }

        uint flags = ReadUInt32(stream, path);
        bool quantisedValues = (flags & 1) != 0;

        int featureCount = (int)ReadUInt32(stream, path);
        int targetCount = (int)ReadUInt32(stream, path);

        var nTrees = new int[targetCount];
        var offsets = new int[targetCount][];
        var feature = new byte[targetCount][];
        var threshold = new float[targetCount][];
        var left = new int[targetCount][];
        var right = new int[targetCount][];
        var value = new float[targetCount][];
        var valueQ = new ushort[targetCount][];
        var valueScale = new float[targetCount];
        var baseValue = new float[targetCount];

        for (int t = 0; t < targetCount; t++)
        {
            baseValue[t] = ReadSingle(stream, path);
            valueScale[t] = ReadSingle(stream, path);
            int trees = (int)ReadUInt32(stream, path);
            int nodes = (int)ReadUInt32(stream, path);
            nTrees[t] = trees;

            offsets[t] = ReadInt32(stream, path, trees + 1);
            feature[t] = ReadBytes(stream, path, nodes);
            threshold[t] = ReadSingle(stream, path, nodes);
            left[t] = ReadInt32(stream, path, nodes);
            right[t] = ReadInt32(stream, path, nodes);
            if (quantisedValues)
            {
                valueQ[t] = ReadUInt16(stream, path, nodes);
            }
            else
            {
                value[t] = ReadSingle(stream, path, nodes);
            }
        }

        if (stream.Position != stream.Length)
        {
            throw new InvalidDataException(
                $"model.bin has {stream.Length - stream.Position} trailing byte(s); the format is out of sync.");
        }

        return new TreeEnsembleModel(
            featureCount, nTrees, offsets, feature, threshold, left, right,
            value, valueQ, valueScale, baseValue);
    }

    /// <summary>
    /// Evaluates every target for one feature vector into <paramref name="outputs"/>
    /// (index 0 = distance_m, index 1 = duration_s, matching the export order).
    /// Branch-light and allocation-free: this is the per-request hot path.
    /// </summary>
    public void Predict(ReadOnlySpan<float> features, Span<float> outputs)
    {
        if (features.Length != FeatureCount) throw new ArgumentException("feature count mismatch", nameof(features));
        if (outputs.Length < TargetCount) throw new ArgumentException("output span too small", nameof(outputs));

        for (int t = 0; t < TargetCount; t++)
        {
            float accumulator = _baseValue[t];
            int[] offsets = _offsets[t];
            byte[] feat = _feature[t];
            float[] thr = _threshold[t];
            int[] lo = _left[t];
            int[] hi = _right[t];
            int trees = _nTrees[t];
            float[] val = _value[t];
            ushort[] valQ = _valueQ[t];
            float scale = _valueScale[t];

            if (val is not null)
            {
                for (int k = 0; k < trees; k++)
                {
                    int i = offsets[k];
                    while (lo[i] >= 0)
                    {
                        i = features[feat[i]] <= thr[i] ? lo[i] : hi[i];
                    }
                    accumulator += val[i];
                }
            }
            else
            {
                for (int k = 0; k < trees; k++)
                {
                    int i = offsets[k];
                    while (lo[i] >= 0)
                    {
                        i = features[feat[i]] <= thr[i] ? lo[i] : hi[i];
                    }
                    accumulator += (valQ[i] - U16Offset) * scale;
                }
            }
            outputs[t] = accumulator;
        }
    }

    /// <summary>
    /// Fills <paramref name="destination"/> completely, or throws. A short read means a
    /// truncated file, which the old whole-file <c>File.ReadAllBytes</c> path could not
    /// distinguish from a valid smaller model.
    /// </summary>
    private static void ReadExactly(Stream stream, Span<byte> destination, string path)
    {
        while (!destination.IsEmpty)
        {
            int read = stream.Read(destination);
            if (read <= 0)
            {
                throw new InvalidDataException($"'{path}' is truncated; the model is incomplete.");
            }
            destination = destination[read..];
        }
    }

    private static uint ReadUInt32(Stream stream, string path)
    {
        Span<byte> bytes = stackalloc byte[4];
        ReadExactly(stream, bytes, path);
        return BinaryPrimitives.ReadUInt32LittleEndian(bytes);
    }

    private static float ReadSingle(Stream stream, string path)
    {
        Span<byte> bytes = stackalloc byte[4];
        ReadExactly(stream, bytes, path);
        return BinaryPrimitives.ReadSingleLittleEndian(bytes);
    }

    // The bulk readers go straight into the final typed array, so the only copy of the
    // table is the one the interpreter actually walks. Little-endian, as the format is.
    private static int[] ReadInt32(Stream stream, string path, int count)
    {
        var array = new int[count];
        ReadExactly(stream, MemoryMarshal.AsBytes(array.AsSpan()), path);
        return array;
    }

    private static float[] ReadSingle(Stream stream, string path, int count)
    {
        var array = new float[count];
        ReadExactly(stream, MemoryMarshal.AsBytes(array.AsSpan()), path);
        return array;
    }

    private static ushort[] ReadUInt16(Stream stream, string path, int count)
    {
        var array = new ushort[count];
        ReadExactly(stream, MemoryMarshal.AsBytes(array.AsSpan()), path);
        return array;
    }

    private static byte[] ReadBytes(Stream stream, string path, int count)
    {
        var array = new byte[count];
        ReadExactly(stream, array, path);
        return array;
    }
}
