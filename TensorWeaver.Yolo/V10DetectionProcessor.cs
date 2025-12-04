using System.Collections.ObjectModel;
using Collections.Pooled;
using CommunityToolkit.Diagnostics;
using TensorWeaver.OutputData;
using TensorWeaver.OutputProcessing;

namespace TensorWeaver.Yolo;

public sealed class V10DetectionProcessor : BoundedOutputProcessor<ReadOnlyCollection<Detection>>, IDisposable
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

	public V10DetectionProcessor(YoloMetadata metadata)
	{
		_imageSize = metadata.ImageSize;
		_buffer = new PooledList<Detection>();
		_wrappedBuffer = new ReadOnlyCollection<Detection>(_buffer);
	}

	public ReadOnlyCollection<Detection> Process(RawOutput output)
	{
		const int boundingCoordinates = 4;
		var tensor = output.Tensors[0];
		var stride = tensor.Strides[1];
		var stride2 = tensor.Strides[2];
		var detectionsCount = tensor.Dimensions[1];
		var tensorSpan = tensor.Buffer.Span;
		PrepareBuffer();
		for (ushort detectionIndex = 0; detectionIndex < detectionsCount; detectionIndex++)
		{
			var boxOffset = detectionIndex * stride;
			var confidence = tensorSpan[boxOffset + boundingCoordinates * stride2];
			if (confidence <= MinimumConfidence)
				continue;
			var classId = (ushort)tensorSpan[boxOffset + 5 * stride2];
			var bounding = ProcessBounding(tensorSpan, detectionIndex, stride);
			if (bounding.Area == 0)
				continue;
			Classification classification = new(classId, confidence);
			Detection detection = new(classification, bounding, detectionIndex);
			_buffer.Add(detection);
		}
		return _wrappedBuffer;
	}

	public void Dispose()
	{
		_buffer.Dispose();
		_nonMaxSuppressor.Dispose();
	}

	private readonly PooledList<Detection> _buffer;
	private readonly ReadOnlyCollection<Detection> _wrappedBuffer;
	private readonly NonMaxSuppressor _nonMaxSuppressor = new();
	private readonly Vector2D<int> _imageSize;
	private float _minimumConfidence = 0.3f;

	private void PrepareBuffer()
	{
		_buffer.Clear();
	}

	private Bounding ProcessBounding(ReadOnlySpan<float> data, int index, int stride)
	{
		var offset = index * stride;
		var left = data[offset + 0];
		var top = data[offset + 1];
		var right = data[offset + 2];
		var bottom = data[offset + 3];
		return new Bounding(left, top, right, bottom) / _imageSize;
	}
}