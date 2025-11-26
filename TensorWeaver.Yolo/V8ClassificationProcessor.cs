using System.Collections.ObjectModel;
using Collections.Pooled;
using CommunityToolkit.Diagnostics;
using TensorWeaver.OutputData;
using TensorWeaver.OutputProcessing;

namespace TensorWeaver.Yolo;

public sealed class V8ClassificationProcessor : OutputProcessor<ReadOnlyCollection<Classification>>
{
	public static V8ClassificationProcessor Instance { get; } = new();

	public float MinimumConfidence
	{
		get => _minimumConfidence;
		set
		{
			Guard.IsInRange(value, 0, 1);
			_minimumConfidence = value;
		}
	}

	public V8ClassificationProcessor()
	{
		_buffer = new PooledList<Classification>();
		_wrappedBuffer = new ReadOnlyCollection<Classification>(_buffer);
	}

	public ReadOnlyCollection<Classification> Process(RawOutput output)
	{
		var span = output.Output0.Buffer.Span;
		PrepareBuffer();
		for (ushort classIndex = 0; classIndex < span.Length; classIndex++)
		{
			var confidence = span[classIndex];
			if (confidence < MinimumConfidence)
				continue;
			Classification classification = new(classIndex, confidence);
			_buffer.Add(classification);
		}
		_buffer.Sort(ReversePredictionConfidenceComparer.Instance);
		return _wrappedBuffer;
	}

	private readonly ReadOnlyCollection<Classification> _wrappedBuffer;
	private readonly PooledList<Classification> _buffer;
	private float _minimumConfidence;

	private void PrepareBuffer()
	{
		_buffer.Clear();
	}
}