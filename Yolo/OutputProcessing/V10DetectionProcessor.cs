using Collections.Pooled;
using CommunityToolkit.Diagnostics;

namespace Yolo.OutputProcessing;

public sealed class V10DetectionProcessor : BoundedOutputProcessor<Detection>
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
		get => _nonMaxSuppressor.MaximumIoU;
		set => _nonMaxSuppressor.MaximumIoU = value;
	}

	public V10DetectionProcessor(Metadata metadata)
	{
		_imageSize = metadata.ImageSize;
	}

	public IReadOnlyList<Detection> Process(RawOutput output)
	{
		var tensor = output.Output0;
		var stride1 = tensor.Strides[1];
		var stride2 = tensor.Strides[2];

		var boxesCount = tensor.Dimensions[1];
		var boxes = new PooledList<Detection>();
		var tensorSpan = tensor.Buffer.Span;

		for (var index = 0; index < boxesCount; index++)
		{
			var boxOffset = index * stride1;

			var confidence = tensorSpan[boxOffset + 4 * stride2];

			if (confidence <= MinimumConfidence)
			{
				continue;
			}

			var classId = (int)tensorSpan[boxOffset + 5 * stride2];
			var box = ProcessBounding(tensorSpan, index, stride1);

			if (box.Width == 0 || box.Height == 0)
			{
				continue;
			}

			boxes.Add(new Detection(new Classification((ushort)classId, confidence), box, (ushort)index));
		}
		
		boxes.Sort(ReverseDetectionClassificationConfidenceComparer.Instance);

		return _nonMaxSuppressor.Suppress(boxes);
	}

	private readonly NonMaxSuppressor _nonMaxSuppressor = new();
	private readonly Vector2D<int> _imageSize;
	private float _minimumConfidence = 0.3f;

	private Bounding ProcessBounding(ReadOnlySpan<float> data, int index, int stride)
	{
		var boxOffset = index * stride;

		var x = data[boxOffset + 0];
		var y = data[boxOffset + 1];
		var w = data[boxOffset + 2];
		var h = data[boxOffset + 3];

		return new Bounding(x, y, w, h) / _imageSize;


		/*var xCenter = data[0 + index];
		var yCenter = data[1 * stride + index];
		var width = data[2 * stride + index];
		var height = data[3 * stride + index];

		var left = xCenter - width / 2;
		var top = yCenter - height / 2;
		var right = xCenter + width / 2;
		var bottom = yCenter + height / 2;

		return new Bounding(left, top, right, bottom) / _imageSize;*/
	}
}