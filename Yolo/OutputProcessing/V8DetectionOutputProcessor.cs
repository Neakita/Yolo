using Collections.Pooled;

namespace Yolo.OutputProcessing;

public sealed class V8DetectionOutputProcessor : OutputProcessor<Detection>
{
	public float MaximumIoU
	{
		get => _suppressor.MaximumIoU;
		set => _suppressor.MaximumIoU = value;
	}

	public override IEnumerable<Detection> Process(RawOutput output)
	{
		var tensor = output.Output0;
		var classStride = tensor.Strides[1];
		var detectionsCount = tensor.Dimensions[2];
		const int boundingCoordinates = 4;
		var boundingStride = tensor.Dimensions[1];
		var classesCount = boundingStride - boundingCoordinates;
		using PooledList<Detection> detections = new();
		var tensorSpan = tensor.Buffer.Span;
		for (var detectionIndex = 0; detectionIndex < detectionsCount; detectionIndex++)
		for (ushort classIndex = 0; classIndex < classesCount; classIndex++)
		{
			var confidence = tensorSpan[(classIndex + boundingCoordinates) * classStride + detectionIndex];
			if (confidence < MinimumConfidence)
				continue;
			var bounding = ProcessBounding(tensorSpan, detectionIndex, classStride);
			if (bounding.Width == 0 || bounding.Height == 0)
				continue;
			detections.Add(new Detection(new Classification(classIndex, confidence), bounding));
		}
		detections.Sort(ReverseDetectionClassificationConfidenceComparer.Instance);
		foreach (var detection in _suppressor.Suppress(detections))
			yield return detection;
	}

	private readonly NonMaxSuppressor _suppressor = new();

	private static Bounding ProcessBounding(ReadOnlySpan<float> data, int index, int stride)
	{
		var xCenter = data[0 + index];
		var yCenter = data[1 * stride + index];
		var width = data[2 * stride + index];
		var height = data[3 * stride + index];

		var left = (int)(xCenter - width / 2);
		var top = (int)(yCenter - height / 2);
		var right = (int)(xCenter + width / 2);
		var bottom = (int)(yCenter + height / 2);

		return new Bounding(left, top, right, bottom);
	}
}