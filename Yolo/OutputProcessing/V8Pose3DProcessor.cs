using System.Collections.Immutable;
using Collections.Pooled;
using CommunityToolkit.Diagnostics;

namespace Yolo.OutputProcessing;

public sealed class V8Pose3DProcessor : BoundedOutputProcessor<Pose3D>
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

	public V8Pose3DProcessor(Predictor predictor, BoundedOutputProcessor<Pose> poseProcessor)
	{
		Guard.IsNotNull(predictor.PoserMetadata);
		Guard.IsEqualTo<byte>(predictor.PoserMetadata.KeyPointsDimensions, 3);
		_metadata = predictor.Metadata;
		_poseProcessor = poseProcessor;
	}

	public IReadOnlyList<Pose3D> Process(RawOutput output)
	{
		const int boundingCoordinatesCount = 4;
		const int keyPointDimensions = 3;
		var tensor = output.Output0;
		var stride = tensor.Strides[1];
		var poses = _poseProcessor.Process(output);
		PooledList<Pose3D> poses3D = new(poses.Count);
		foreach (var pose in poses)
		{
			PooledList<KeyPoint3D> keyPoints3D = new(pose.KeyPoints.Count);
			for (byte keyPointIndex = 0; keyPointIndex < pose.KeyPoints.Count; keyPointIndex++)
			{
				var offset = keyPointIndex * keyPointDimensions + boundingCoordinatesCount + _metadata.ClassesNames.Length;
				var keyPointConfidence = tensor.Buffer.Span[(offset + 2) * stride + pose.Detection.Index];
				KeyPoint keyPoint = pose.KeyPoints[keyPointIndex];
				KeyPoint3D keyPoint3D = new(keyPoint.Position, keyPointConfidence);
				keyPoints3D.Add(keyPoint3D);
			}
			if (pose.KeyPoints is IDisposable keyPointsDisposable)
				keyPointsDisposable.Dispose();
			Pose3D pose3D = new(pose.Detection, keyPoints3D);
			poses3D.Add(pose3D);
		}
		if (poses is IDisposable posesDisposable)
			posesDisposable.Dispose();
		return poses3D;
	}

	private readonly Metadata _metadata;
	private readonly BoundedOutputProcessor<Pose> _poseProcessor;
}