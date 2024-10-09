using Collections.Pooled;
using CommunityToolkit.Diagnostics;

namespace Yolo.OutputProcessing;

public sealed class V8DetectionProcessor : BoundedOutputProcessor<Detection>, IDisposable
{
	public float MinimumConfidence
	{
		get => _minimumConfidence;
		set
		{
			Guard.IsInRange(value, 0, 1);
			_minimumConfidence = value;
		}
	}

	public float MaximumIoU
	{
		get => _suppressor.MaximumIoU;
		set => _suppressor.MaximumIoU = value;
	}

	public V8DetectionProcessor(Metadata metadata)
	{
		_classesCount = (ushort)metadata.ClassesNames.Length;
		_imageSize = metadata.ImageSize;
	}

	public IReadOnlyList<Detection> Process(RawOutput output)
	{
		const int boundingCoordinates = 4;
		var tensor = output.Output0;
		var stride = tensor.Strides[1];
		var detectionsCount = tensor.Dimensions[2];
		var tensorSpan = tensor.Buffer.Span;
		PrepareBuffer();
		for (ushort detectionIndex = 0; detectionIndex < detectionsCount; detectionIndex++)
		for (ushort classIndex = 0; classIndex < _classesCount; classIndex++)
		{
			var confidence = tensorSpan[(classIndex + boundingCoordinates) * stride + detectionIndex];
			if (confidence < MinimumConfidence)
				continue;
			var bounding = ProcessBounding(tensorSpan, detectionIndex, stride);
			if (bounding.Area == 0)
				continue;
			Classification classification = new(classIndex, confidence);
			Detection detection = new(classification, bounding, detectionIndex);
			_buffer.Add(detection);
		}
		_buffer.Sort(ReverseDetectionClassificationConfidenceComparer.Instance);
		return _suppressor.Suppress(_buffer);
	}

	public void Dispose()
	{
		_suppressor.Dispose();
		_buffer.Dispose();
	}

	private readonly ushort _classesCount;
	private readonly NonMaxSuppressor _suppressor = new();
	private readonly PooledList<Detection> _buffer = new();
	private readonly Vector2D<int> _imageSize;
	private float _minimumConfidence = 0.3f;

	private Bounding ProcessBounding(ReadOnlySpan<float> data, int index, int stride)
	{
		var xCenter = data[index];
		var yCenter = data[index + stride];
		var width = data[index + stride * 2];
		var height = data[index + stride * 3];

		var left = xCenter - width / 2;
		var top = yCenter - height / 2;
		var right = xCenter + width / 2;
		var bottom = yCenter + height / 2;

		return new Bounding(left, top, right, bottom) / _imageSize;
	}

	private void PrepareBuffer()
	{
		_buffer.Clear();
	}
}