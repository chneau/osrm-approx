using System.Buffers.Binary;

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
        byte[] buffer = File.ReadAllBytes(path);
        int pos = 0;

        if (buffer.Length < 16 || !buffer.AsSpan(0, 4).SequenceEqual(Magic))
        {
            throw new InvalidDataException($"'{path}' is not an OSRT tree ensemble (bad magic).");
        }
        pos = 4;

        uint version = ReadUInt32(buffer, ref pos);
        if (version != 1)
        {
            throw new InvalidDataException($"Unsupported model.bin version {version} (expected 1).");
        }

        int featureCount = (int)ReadUInt32(buffer, ref pos);
        int targetCount = (int)ReadUInt32(buffer, ref pos);

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
            baseValue[t] = ReadSingle(buffer, ref pos);
            int trees = (int)ReadUInt32(buffer, ref pos);
            int nodes = (int)ReadUInt32(buffer, ref pos);
            nTrees[t] = trees;

            offsets[t] = ReadInt32(buffer, ref pos, trees + 1);
            feature[t] = ReadInt32(buffer, ref pos, nodes);
            threshold[t] = ReadSingle(buffer, ref pos, nodes);
            left[t] = ReadInt32(buffer, ref pos, nodes);
            right[t] = ReadInt32(buffer, ref pos, nodes);
            value[t] = ReadSingle(buffer, ref pos, nodes);
            isLeaf[t] = ReadBytes(buffer, ref pos, nodes);
            _ = ReadBytes(buffer, ref pos, nodes); // default_left: unused (no NaN features)
        }

        if (pos != buffer.Length)
        {
            throw new InvalidDataException(
                $"model.bin has {buffer.Length - pos} trailing byte(s); the format is out of sync.");
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

    private static uint ReadUInt32(byte[] buffer, ref int pos)
    {
        uint v = BinaryPrimitives.ReadUInt32LittleEndian(buffer.AsSpan(pos));
        pos += 4;
        return v;
    }

    private static float ReadSingle(byte[] buffer, ref int pos)
    {
        float v = BinaryPrimitives.ReadSingleLittleEndian(buffer.AsSpan(pos));
        pos += 4;
        return v;
    }

    private static int[] ReadInt32(byte[] buffer, ref int pos, int count)
    {
        var array = new int[count];
        Buffer.BlockCopy(buffer, pos, array, 0, count * sizeof(int));
        pos += count * sizeof(int);
        return array;
    }

    private static float[] ReadSingle(byte[] buffer, ref int pos, int count)
    {
        var array = new float[count];
        Buffer.BlockCopy(buffer, pos, array, 0, count * sizeof(float));
        pos += count * sizeof(float);
        return array;
    }

    private static byte[] ReadBytes(byte[] buffer, ref int pos, int count)
    {
        var array = new byte[count];
        Buffer.BlockCopy(buffer, pos, array, 0, count);
        pos += count;
        return array;
    }
}
