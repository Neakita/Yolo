using System.Collections.ObjectModel;
using Collections.Pooled;
using CommunityToolkit.Diagnostics;
using Microsoft.ML.OnnxRuntime;
using TensorWeaver.Metadata;
using TensorWeaver.OutputData;
using TensorWeaver.OutputProcessing;

namespace TensorWeaver.Yolo;

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

	public V8Pose3DProcessor(InferenceSession session)
	{
		_metadata = YoloMetadata.Parse(session);
		_poseProcessor = new V8PoseProcessor(session);
		_poserMetadata = PoserMetadata.Parse(session.ModelMetadata.CustomMetadataMap);
		Guard.IsEqualTo<byte>(_poserMetadata.KeyPointsDimensions, 3);
		_posesBuffer = new PooledList<Pose3D>();
		_wrappedPosesBuffer = new ReadOnlyCollection<Pose3D>(_posesBuffer);
	}

	public ReadOnlyCollection<Pose3D> Process(RawOutput output)
	{
		const int boundingCoordinatesCount = 4;
		const int keyPointDimensions = 3;
		var tensor = output.Output0;
		var stride = tensor.Strides[1];
		var poses = _poseProcessor.Process(output);
		for (var i = 0; i < poses.Count; i++)
		{
			var pose = poses[i];
			GetKeyPointBuffers(i, out var keyPointsBuffer, out var wrappedKeyPointsBuffer);
			for (byte keyPointIndex = 0; keyPointIndex < pose.KeyPoints.Count; keyPointIndex++)
			{
				var offset = keyPointIndex * keyPointDimensions + boundingCoordinatesCount + _metadata.ClassesNames.Length;
				var keyPointConfidence = tensor.Buffer.Span[(offset + 2) * stride + pose.Detection.Index];
				KeyPoint keyPoint = pose.KeyPoints[keyPointIndex];
				KeyPoint3D keyPoint3D = new(keyPoint.Position, keyPointConfidence);
				keyPointsBuffer.Add(keyPoint3D);
			}
			Pose3D pose3D = new(pose.Detection, wrappedKeyPointsBuffer);
			_posesBuffer.Add(pose3D);
		}
		return _wrappedPosesBuffer;
	}

	public void Dispose()
	{
		_poseProcessor.Dispose();
		_posesBuffer.Dispose();
		_keyPointBuffers.Dispose();
	}

	private readonly YoloMetadata _metadata;
	private readonly V8PoseProcessor _poseProcessor;
	private readonly PooledList<Pose3D> _posesBuffer;
	private readonly ReadOnlyCollection<Pose3D> _wrappedPosesBuffer;
	private readonly PooledList<PooledList<KeyPoint3D>> _keyPointBuffers = new();
	private readonly PooledList<ReadOnlyCollection<KeyPoint3D>> _keyPointWrappedBuffers = new();
	private readonly PoserMetadata _poserMetadata;

	private void GetKeyPointBuffers(
		int index,
		out PooledList<KeyPoint3D> buffer,
		out ReadOnlyCollection<KeyPoint3D> wrappedBuffer)
	{
		if (index == _keyPointBuffers.Count)
		{
			buffer = new PooledList<KeyPoint3D>(_poserMetadata.KeyPointsCount);
			wrappedBuffer = new ReadOnlyCollection<KeyPoint3D>(buffer);
			_keyPointBuffers.Add(buffer);
			_keyPointWrappedBuffers.Add(wrappedBuffer);
		}
		buffer = _keyPointBuffers[index];
		wrappedBuffer = _keyPointWrappedBuffers[index];
		buffer.Clear();
	}
}