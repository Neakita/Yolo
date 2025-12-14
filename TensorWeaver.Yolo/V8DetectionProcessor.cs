using CommunityToolkit.Diagnostics;
using TensorWeaver.OutputData;
using TensorWeaver.OutputProcessing;

namespace TensorWeaver.Yolo;

public sealed class V8DetectionProcessor : OutputProcessor<List<Detection>>
{
	public float MinimumConfidence
	{
		get;
		set
		{
			Guard.IsInRange(value, 0, 1);
			field = value;
		}
	} = 0.3f;

	public float MaximumIoU
	{
		get => _suppressor.MaximumIoU;
		set => _suppressor.MaximumIoU = value;
	}

	public V8DetectionProcessor(YoloMetadata metadata)
	{
		_classesCount = (ushort)metadata.ClassesNames.Length;
		_imageSize = metadata.ImageSize;
	}

	public List<Detection> Process(RawOutput output)
	{
		const int boundingCoordinates = 4;
		var tensor = output.Tensors[0];
		var stride = tensor.Strides[1];
		var detectionsCount = tensor.Dimensions[2];
		var tensorSpan = tensor.Buffer.Span;
		var detections = new List<Detection>();
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
			detections.Add(detection);
		}
		detections.Sort(ReverseDetectionClassificationConfidenceComparer.Instance);
		return _suppressor.Suppress(detections);
	}

	private readonly ushort _classesCount;
	private readonly NonMaxSuppressor _suppressor = new();
	private readonly Vector2D<int> _imageSize;

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
}