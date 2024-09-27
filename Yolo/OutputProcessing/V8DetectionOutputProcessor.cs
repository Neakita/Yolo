using System.Collections.Immutable;

namespace Yolo.OutputProcessing;

public sealed class V8DetectionOutputProcessor : OutputProcessor<Detection>
{
	public float IoUThreshold { get; set; } = 0.45f;

	public override ImmutableArray<Detection> Process(RawOutput output)
	{
		var tensor = output.Output0;
		var classStride = tensor.Strides[1];
		var detectionsCount = tensor.Dimensions[2];
		const int boundingCoordinates = 4;
		var boundingStride = tensor.Dimensions[1];
		var classesCount = boundingStride - boundingCoordinates;
		Span<ValueDetection> detections = stackalloc ValueDetection[detectionsCount];
		var boxesIndex = 0;
		var tensorSpan = tensor.Buffer.Span;
		for (var boxIndex = 0; boxIndex < detectionsCount; boxIndex++)
		for (ushort classIndex = 0; classIndex < classesCount; classIndex++)
		{
			var confidence = tensorSpan[(classIndex + 4) * classStride + boxIndex];
			if (confidence < MinimumConfidence)
				continue;
			var bounding = ProcessBounding(tensorSpan, boxIndex, classStride);
			if (bounding.Width == 0 || bounding.Height == 0)
				continue;
			detections[boxesIndex++] = new ValueDetection(bounding, classIndex, confidence);
		}
		return NonMaxSuppressor.SuppressAndCombine(detections[..boxesIndex], IoUThreshold);
	}

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