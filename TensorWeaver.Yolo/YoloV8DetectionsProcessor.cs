using Microsoft.ML.OnnxRuntime.Tensors;
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
		_classesCount = metadata.ClassesNames.Length;
		_imageSize = metadata.ImageSize;
	}

	public YoloV8DetectionsProcessor(int classesCount, Vector2D<int> imageSize)
	{
		_classesCount = classesCount;
		_imageSize = imageSize;
	}

	public List<Detection> Process(RawOutput output)
	{
		var tensor = output.Tensors[0];
		var detectionsCount = tensor.Dimensions[2];
		var detections = new List<Detection>();
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
			detections.Add(detection);
		}
		detections.Sort(ReverseDetectionClassificationConfidenceComparer.Instance);
		return _suppressor.Suppress(detections);
	}

	private const int BatchIndex = 0;
	private const int FirstClassIndex = 4;
	private readonly int _classesCount;
	private readonly NonMaxSuppressor _suppressor = new();
	private readonly Vector2D<int> _imageSize;

	private Bounding ProcessBounding(DenseTensor<float> data, int detectionIndex)
	{
		var xCenter = data[BatchIndex, 0, detectionIndex];
		var yCenter = data[BatchIndex, 1, detectionIndex];
		var width = data[BatchIndex, 2, detectionIndex];
		var height = data[BatchIndex, 3, detectionIndex];
		return Bounding.FromPoint(xCenter, yCenter, width, height) / _imageSize;
	}
}