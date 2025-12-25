using CommunityToolkit.HighPerformance;
using Microsoft.ML.OnnxRuntime.Tensors;
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
		var tensor = output.Tensors[0];
		var detectionsCount = tensor.Dimensions[2];
		_buffer.Clear();
		for (int detectionIndex = 0; detectionIndex < detectionsCount; detectionIndex++)
		for (int classIndex = 0; classIndex < _classesCount; classIndex++)
		{
			var confidence = tensor[BatchIndex, FirstClassIndex + classIndex, detectionIndex];
			if (confidence < MinimumConfidence)
				continue;
			var bounding = ProcessBounding(tensor, detectionIndex);
			if (bounding.Area == 0)
				continue;
			var classification = new Classification((ushort)classIndex, confidence);
			var detection = new Detection(classification, bounding, (ushort)detectionIndex);
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

	private const int BatchIndex = 0;
	private const int FirstClassIndex = 4;
	private readonly ushort _classesCount;
	private readonly NonMaxSpanSuppressor _suppressor = new();
	private readonly Vector2D<int> _imageSize;
	private readonly List<Detection> _buffer = new();
	private readonly IComparer<Detection>? _detectionsComparer;
	private readonly Comparison<Detection> _confidenceReverseComparison = static (x, y) => y.Confidence.CompareTo(x.Confidence);

	private Bounding ProcessBounding(DenseTensor<float> data, int detectionIndex)
	{
		var xCenter = data[BatchIndex, 0, detectionIndex];
		var yCenter = data[BatchIndex, 1, detectionIndex];
		var width = data[BatchIndex, 2, detectionIndex];
		var height = data[BatchIndex, 3, detectionIndex];
		return Bounding.FromPoint(xCenter, yCenter, width, height) / _imageSize;
	}
}