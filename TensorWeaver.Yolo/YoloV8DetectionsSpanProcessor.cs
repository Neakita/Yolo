using CommunityToolkit.HighPerformance;
using TensorWeaver.OutputData;
using TensorWeaver.OutputProcessing;

namespace TensorWeaver.Yolo;

public sealed class YoloV8DetectionsSpanProcessor : OutputSpanProcessor<Detection>
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

	public YoloV8DetectionsSpanProcessor(YoloMetadata metadata, IComparer<Detection>? detectionsComparer = null)
	{
		_classesCount = (ushort)metadata.ClassesNames.Length;
		_imageSize = metadata.ImageSize;
		_detectionsComparer = detectionsComparer;
	}

	public YoloV8DetectionsSpanProcessor(ushort classesCount, Vector2D<int> imageSize, IComparer<Detection>? detectionsComparer = null)
	{
		_classesCount = classesCount;
		_imageSize = imageSize;
		_detectionsComparer = detectionsComparer;
	}

	public int Process(RawOutput output, Span<Detection> target)
	{
		const int boundingCoordinates = 4;
		var tensor = output.Tensors[0];
		var stride = tensor.Strides[1];
		var detectionsCount = tensor.Dimensions[2];
		var tensorSpan = tensor.Buffer.Span;
		_buffer.Clear();
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
		_buffer.Sort(_confidenceReverseComparison);
		var bufferSpan = _buffer.AsSpan();
		var passedDetectionsCount = _suppressor.Suppress(bufferSpan, bufferSpan);
		var passedDetections = bufferSpan[..passedDetectionsCount];
		if (_detectionsComparer != null)
			passedDetections.Sort(_detectionsComparer);
		if (target.Length < passedDetections.Length)
			passedDetections = passedDetections[..target.Length];
		passedDetections.CopyTo(target);
		return passedDetectionsCount;
	}

	private readonly ushort _classesCount;
	private readonly NonMaxSpanSuppressor _suppressor = new();
	private readonly Vector2D<int> _imageSize;
	private readonly List<Detection> _buffer = new();
	private readonly IComparer<Detection>? _detectionsComparer;
	private readonly Comparison<Detection> _confidenceReverseComparison = static (x, y) => y.Confidence.CompareTo(x.Confidence);

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