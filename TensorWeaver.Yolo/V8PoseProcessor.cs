using System.Collections.ObjectModel;
using Collections.Pooled;
using Microsoft.ML.OnnxRuntime;
using TensorWeaver.OutputData;
using TensorWeaver.OutputProcessing;

namespace TensorWeaver.Yolo;

public sealed class V8PoseProcessor : BoundedOutputProcessor<ReadOnlyCollection<Pose>>, IDisposable
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

	public V8PoseProcessor(InferenceSession session)
	{
		_metadata = YoloMetadata.Parse(session);
		_poserMetadata = PoserMetadata.Parse(session.ModelMetadata.CustomMetadataMap);
		_detectionProcessor = new V8DetectionProcessor(_metadata);
		_posesBuffer = new PooledList<Pose>();
		_wrappedPosesBuffer = new ReadOnlyCollection<Pose>(_posesBuffer);
	}

	public ReadOnlyCollection<Pose> Process(RawOutput output)
	{
		const int boundingCoordinates = 4;
		var tensor = output.Output0;
		var stride = tensor.Strides[1];
		var detections = _detectionProcessor.Process(output);
		PreparePosesBuffer();
		for (var i = 0; i < detections.Count; i++)
		{
			var detection = detections[i];
			GetKeyPointsBuffer(i, out var keyPointsBuffer, out var wrappedKeyPointsBuffer);
			for (byte keyPointIndex = 0; keyPointIndex < _poserMetadata.KeyPointsCount; keyPointIndex++)
			{
				var offset = keyPointIndex * _poserMetadata.KeyPointsDimensions + boundingCoordinates +
				             _metadata.ClassesNames.Length;
				var pointX = (int)tensor.Buffer.Span[offset * stride + detection.Index];
				var pointY = (int)tensor.Buffer.Span[(offset + 1) * stride + detection.Index];
				Vector2D<float> position = new(pointX, pointY);
				position /= _metadata.ImageSize.ToSingle();
				KeyPoint keyPoint = new(position);
				keyPointsBuffer.Add(keyPoint);
			}
			Pose pose = new(detection, wrappedKeyPointsBuffer);
			_posesBuffer.Add(pose);
		}
		return _wrappedPosesBuffer;
	}

	public void Dispose()
	{
		_detectionProcessor.Dispose();
		_posesBuffer.Dispose();
		foreach (var keyPointBuffer in _keyPointBuffers)
			keyPointBuffer.Dispose();
		_keyPointBuffers.Dispose();
	}

	private readonly YoloMetadata _metadata;
	private readonly PoserMetadata _poserMetadata;
	private readonly V8DetectionProcessor _detectionProcessor;
	private readonly PooledList<Pose> _posesBuffer;
	private readonly ReadOnlyCollection<Pose> _wrappedPosesBuffer;
	private readonly PooledList<PooledList<KeyPoint>> _keyPointBuffers = new();
	private readonly PooledList<ReadOnlyCollection<KeyPoint>> _keyPointWrappedBuffers = new();

	private void PreparePosesBuffer()
	{
		_posesBuffer.Clear();
	}

	private void GetKeyPointsBuffer(
		int index,
		out PooledList<KeyPoint> buffer,
		out ReadOnlyCollection<KeyPoint> wrappedBuffer)
	{
		if (index == _keyPointBuffers.Count)
		{
			buffer = new PooledList<KeyPoint>(_poserMetadata.KeyPointsCount);
			wrappedBuffer = new ReadOnlyCollection<KeyPoint>(buffer);
			_keyPointBuffers.Add(buffer);
			_keyPointWrappedBuffers.Add(wrappedBuffer);
		}
		buffer = _keyPointBuffers[index];
		wrappedBuffer = _keyPointWrappedBuffers[index];
		buffer.Clear();
	}
}