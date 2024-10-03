using Collections.Pooled;
using CommunityToolkit.Diagnostics;

namespace Yolo.OutputProcessing;

public sealed class V8DetectionProcessor : BoundedOutputProcessor<Detection>
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
		using PooledList<Detection> detections = new();
		var tensorSpan = tensor.Buffer.Span;
		for (ushort detectionIndex = 0; detectionIndex < detectionsCount; detectionIndex++)
		for (ushort classIndex = 0; classIndex < _classesCount; classIndex++)
		{
			var confidence = tensorSpan[(classIndex + boundingCoordinates) * stride + detectionIndex];
			if (confidence < MinimumConfidence)
				continue;
			var bounding = ProcessBounding(tensorSpan, detectionIndex, stride);
			if (bounding.Width == 0 || bounding.Height == 0)
				continue;
			Classification classification = new(classIndex, confidence);
			Detection detection = new(classification, bounding, detectionIndex);
			detections.Add(detection);
		}
		detections.Sort(ReverseDetectionClassificationConfidenceComparer.Instance);
		if (detections.Count == 0)
			return Array.Empty<Detection>();
		return _suppressor.Suppress(detections);
	}

	private readonly ushort _classesCount;
	private readonly NonMaxSuppressor _suppressor = new();
	private readonly Vector2D<int> _imageSize;
	private float _minimumConfidence = 0.3f;

	private Bounding ProcessBounding(ReadOnlySpan<float> data, int index, int stride)
	{
		var xCenter = data[0 + index];
		var yCenter = data[1 * stride + index];
		var width = data[2 * stride + index];
		var height = data[3 * stride + index];

		var left = xCenter - width / 2;
		var top = yCenter - height / 2;
		var right = xCenter + width / 2;
		var bottom = yCenter + height / 2;

		return new Bounding(left, top, right, bottom) / _imageSize;
	}
}