using TensorWeaver.OutputData;
using TensorWeaver.OutputProcessing;

namespace TensorWeaver.Yolo;

public sealed class YoloV8DetectionsProcessor : OutputProcessor<List<Detection>>
{
	public float MinimumConfidence
	{
		get;
		set
		{
			if (value is <= 0 or >= 1)
				throw new ArgumentOutOfRangeException(nameof(MinimumConfidence), value, $"Value for {MinimumConfidence} should be exclusively between 0 and 1, but was {value}");
			field = value;
		}
	} = 0.5f;

	public float MaximumIoU
	{
		get => _suppressor.MaximumIoU;
		set => _suppressor.MaximumIoU = value;
	}

	public YoloV8DetectionsProcessor(YoloMetadata metadata)
	{
		_classesCount = (byte)metadata.ClassesNames.Length;
		_imageSize = metadata.ImageSize;
	}

	public YoloV8DetectionsProcessor(byte classesCount, Vector2D<int> imageSize)
	{
		_classesCount = classesCount;
		_imageSize = imageSize;
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
		for (byte classIndex = 0; classIndex < _classesCount; classIndex++)
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

	private readonly byte _classesCount;
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