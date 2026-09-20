using System.Buffers.Binary;
using System.Runtime.InteropServices;

/// <summary>
/// E11 (see IMPROVEMENTS.md): a tiny, dependency-free interpreter for the
/// LightGBM tree ensemble, loaded from the compact <c>model.bin</c> emitted by
/// <c>python/export_binary.py</c>. Replaces ONNX Runtime, which was the dominant
/// RSS cost; the trees and leaf values are bit-identical to the ONNX graph.
/// </summary>
/// <remarks>
/// Binary layout (little-endian), written by <c>python/export_binary.py</c>:
/// <code>
///   magic       char[4]  "OSRT"
///   version     uint32
///   n_features  uint32
///   n_targets   uint32
///   per target (graph order: distance_m, duration_s):
///     base_value    float32
///     n_trees       uint32
///     n_nodes       uint32
///     tree_offsets  int32[n_trees+1]   start row of each tree's node block
///     feature       int32[n_nodes]
///     threshold     float32[n_nodes]
///     left          int32[n_nodes]     global row of the x&lt;=threshold child
///     right         int32[n_nodes]     global row of the x&gt;threshold child
///     value         float32[n_nodes]   leaf value (0 for internal nodes)
///     is_leaf       uint8[n_nodes]
///     default_left  uint8[n_nodes]
/// </code>
/// Child ids are already rebased to global rows, so the walk never needs the
/// per-tree offset once the (feature, threshold, left, right) row is chosen.
/// Internal nodes use the BRANCH_LEQ mode: <c>x &lt;= threshold</c> takes the left
/// child (matching ONNX Runtime's TreeEnsembleRegressor semantics).
/// </remarks>
/// <remarks>
/// Loading streams the file rather than reading it whole: each table is read
/// straight into its final array, so no second, full-size copy of the model is ever
/// held. See <see cref="Load"/> for why that matters to RSS.
/// </remarks>
internal sealed class TreeEnsembleModel
{
    private static readonly byte[] Magic = "OSRT"u8.ToArray();

    // Arrays are indexed [target]. Every query walks all targets, so keeping the
    // per-target tables in flat arrays avoids a double indirection per node.
    private readonly int[] _nTrees;
    private readonly int[][] _offsets;
    private readonly int[][] _feature;
    private readonly float[][] _threshold;
    private readonly int[][] _left;
    private readonly int[][] _right;
    private readonly float[][] _value;
    private readonly byte[][] _isLeaf;
    private readonly float[] _baseValue;

    public int FeatureCount { get; }
    public int TargetCount => _nTrees.Length;

    private TreeEnsembleModel(
        int featureCount,
        int[] nTrees,
        int[][] offsets,
        int[][] feature,
        float[][] threshold,
        int[][] left,
        int[][] right,
        float[][] value,
        byte[][] isLeaf,
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
        _isLeaf = isLeaf;
        _baseValue = baseValue;
    }

    public static TreeEnsembleModel Load(string path)
    {
        // Stream the file instead of File.ReadAllBytes. Reading it whole allocated a
        // byte[] the size of the entire table (up to ~18 MB), on top of the ~18 MB of
        // typed node arrays it was then parsed into; being large, that buffer went to
        // the Large Object Heap and stayed resident until a gen2 collection, so the
        // model cost ~1.9x its file size in RSS. Each array is now read straight into
        // its final allocation and the transient copy is never created.
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
        if (version != 1)
        {
            throw new InvalidDataException($"Unsupported model.bin version {version} (expected 1).");
        }

        int featureCount = (int)ReadUInt32(stream, path);
        int targetCount = (int)ReadUInt32(stream, path);

        var nTrees = new int[targetCount];
        var offsets = new int[targetCount][];
        var feature = new int[targetCount][];
        var threshold = new float[targetCount][];
        var left = new int[targetCount][];
        var right = new int[targetCount][];
        var value = new float[targetCount][];
        var isLeaf = new byte[targetCount][];
        var baseValue = new float[targetCount];

        for (int t = 0; t < targetCount; t++)
        {
            baseValue[t] = ReadSingle(stream, path);
            int trees = (int)ReadUInt32(stream, path);
            int nodes = (int)ReadUInt32(stream, path);
            nTrees[t] = trees;

            offsets[t] = ReadInt32(stream, path, trees + 1);
            feature[t] = ReadInt32(stream, path, nodes);
            threshold[t] = ReadSingle(stream, path, nodes);
            left[t] = ReadInt32(stream, path, nodes);
            right[t] = ReadInt32(stream, path, nodes);
            value[t] = ReadSingle(stream, path, nodes);
            isLeaf[t] = ReadBytes(stream, path, nodes);
            SkipBytes(stream, path, nodes); // default_left: unused (no NaN features)
        }

        if (stream.Position != stream.Length)
        {
            throw new InvalidDataException(
                $"model.bin has {stream.Length - stream.Position} trailing byte(s); the format is out of sync.");
        }

        return new TreeEnsembleModel(
            featureCount, nTrees, offsets, feature, threshold, left, right, value, isLeaf, baseValue);
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
            int[] feat = _feature[t];
            float[] thr = _threshold[t];
            int[] lo = _left[t];
            int[] hi = _right[t];
            float[] val = _value[t];
            byte[] leaf = _isLeaf[t];
            int trees = _nTrees[t];

            for (int k = 0; k < trees; k++)
            {
                int i = offsets[k];
                while (leaf[i] == 0)
                {
                    i = features[feat[i]] <= thr[i] ? lo[i] : hi[i];
                }
                accumulator += val[i];
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

    /// <summary>Consumes <paramref name="count"/> bytes without allocating an array.</summary>
    private static void SkipBytes(Stream stream, string path, int count)
    {
        Span<byte> scratch = stackalloc byte[256];
        while (count > 0)
        {
            int chunk = Math.Min(count, scratch.Length);
            ReadExactly(stream, scratch[..chunk], path);
            count -= chunk;
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

    private static byte[] ReadBytes(Stream stream, string path, int count)
    {
        var array = new byte[count];
        ReadExactly(stream, array, path);
        return array;
    }
}
