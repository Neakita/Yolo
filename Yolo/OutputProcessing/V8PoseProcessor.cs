using Collections.Pooled;
using CommunityToolkit.Diagnostics;

namespace Yolo.OutputProcessing;

public sealed class V8PoseProcessor : BoundedOutputProcessor<Pose>, IDisposable
{
	public float MinimumConfidence
	{
		get => _detectionProcessor.MinimumConfidence;
		set => _detectionProcessor.MinimumConfidence = value;
	}

	public float MaximumIoU
	{
		get => _detectionProcessor.MaximumIoU;
		set => _detectionProcessor.MaximumIoU = value;
	}

	public V8PoseProcessor(Predictor predictor)
	{
		_metadata = predictor.Metadata;
		Guard.IsNotNull(predictor.PoserMetadata);
		_poserMetadata = predictor.PoserMetadata;
		_detectionProcessor = new V8DetectionProcessor(predictor.Metadata);
	}

	public IReadOnlyList<Pose> Process(RawOutput output)
	{
		const int boundingCoordinates = 4;
		var tensor = output.Output0;
		var stride = tensor.Strides[1];
		var detections = _detectionProcessor.Process(output);
		PreparePosesBuffer();
		for (var i = 0; i < detections.Count; i++)
		{
			var detection = detections[i];
			PooledList<KeyPoint> keyPoints = PrepareAndGetKeyPointsBuffer(i);
			for (byte keyPointIndex = 0; keyPointIndex < _poserMetadata.KeyPointsCount; keyPointIndex++)
			{
				var offset = keyPointIndex * _poserMetadata.KeyPointsDimensions + boundingCoordinates +
				             _metadata.ClassesNames.Length;
				var pointX = (int)tensor.Buffer.Span[offset * stride + detection.Index];
				var pointY = (int)tensor.Buffer.Span[(offset + 1) * stride + detection.Index];
				Vector2D<float> position = new(pointX, pointY);
				position /= _metadata.ImageSize.ToSingle();
				KeyPoint keyPoint = new(position);
				keyPoints.Add(keyPoint);
			}

			Pose pose = new(detection, keyPoints);
			_posesBuffer.Add(pose);
		}

		return _posesBuffer;
	}

	public void Dispose()
	{
		_detectionProcessor.Dispose();
		_posesBuffer.Dispose();
	}

	private readonly Metadata _metadata;
	private readonly PoserMetadata _poserMetadata;
	private readonly V8DetectionProcessor _detectionProcessor;
	private readonly PooledList<Pose> _posesBuffer = new();
	private readonly PooledList<PooledList<KeyPoint>> _keyPointBuffers = new();

	private void PreparePosesBuffer()
	{
		_posesBuffer.Clear();
	}

	private PooledList<KeyPoint> PrepareAndGetKeyPointsBuffer(int index)
	{
		if (index == _keyPointBuffers.Count)
		{
			PooledList<KeyPoint> newBuffer = new(_poserMetadata.KeyPointsCount);
			_keyPointBuffers.Add(newBuffer);
			return newBuffer;
		}
		var existingBuffer = _keyPointBuffers[index];
		existingBuffer.Clear();
		return existingBuffer;
	}
}