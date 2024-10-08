using Collections.Pooled;
using CommunityToolkit.Diagnostics;

namespace Yolo.OutputProcessing;

public sealed class V8Pose3DProcessor : BoundedOutputProcessor<Pose3D>, IDisposable
{
	public float MinimumConfidence
	{
		get => _poseProcessor.MinimumConfidence;
		set => _poseProcessor.MinimumConfidence = value;
	}

	public float MaximumIoU
	{
		get => _poseProcessor.MaximumIoU;
		set => _poseProcessor.MaximumIoU = value;
	}

	public V8Pose3DProcessor(Predictor predictor)
	{
		Guard.IsNotNull(predictor.PoserMetadata);
		Guard.IsEqualTo<byte>(predictor.PoserMetadata.KeyPointsDimensions, 3);
		_metadata = predictor.Metadata;
		_poseProcessor = new V8PoseProcessor(predictor);
		_poserMetadata = predictor.PoserMetadata;
	}

	public IReadOnlyList<Pose3D> Process(RawOutput output)
	{
		const int boundingCoordinatesCount = 4;
		const int keyPointDimensions = 3;
		var tensor = output.Output0;
		var stride = tensor.Strides[1];
		var poses = _poseProcessor.Process(output);
		for (var i = 0; i < poses.Count; i++)
		{
			var pose = poses[i];
			PooledList<KeyPoint3D> keyPoints3D = GetKeyPointsBuffer(i);
			for (byte keyPointIndex = 0; keyPointIndex < pose.KeyPoints.Count; keyPointIndex++)
			{
				var offset = keyPointIndex * keyPointDimensions + boundingCoordinatesCount + _metadata.ClassesNames.Length;
				var keyPointConfidence = tensor.Buffer.Span[(offset + 2) * stride + pose.Detection.Index];
				KeyPoint keyPoint = pose.KeyPoints[keyPointIndex];
				KeyPoint3D keyPoint3D = new(keyPoint.Position, keyPointConfidence);
				keyPoints3D.Add(keyPoint3D);
			}
			Pose3D pose3D = new(pose.Detection, keyPoints3D);
			_posesBuffer.Add(pose3D);
		}

		return _posesBuffer;
	}

	public void Dispose()
	{
		_poseProcessor.Dispose();
		_posesBuffer.Dispose();
		_keyPointBuffers.Dispose();
	}

	private readonly Metadata _metadata;
	private readonly V8PoseProcessor _poseProcessor;
	private readonly PooledList<Pose3D> _posesBuffer = new();
	private readonly PooledList<PooledList<KeyPoint3D>> _keyPointBuffers = new();
	private readonly PoserMetadata _poserMetadata;

	private PooledList<KeyPoint3D> GetKeyPointsBuffer(int index)
	{
		if (index == _keyPointBuffers.Count)
		{
			PooledList<KeyPoint3D> newBuffer = new(_poserMetadata.KeyPointsCount);
			_keyPointBuffers.Add(newBuffer);
			return newBuffer;
		}
		var existingBuffer = _keyPointBuffers[index];
		existingBuffer.Clear();
		return existingBuffer;
	}
}