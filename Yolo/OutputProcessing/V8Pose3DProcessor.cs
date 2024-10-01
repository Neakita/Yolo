using System.Collections.Immutable;
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

	public IEnumerable<Pose3D> Process(RawOutput output)
	{
		const int boundingCoordinatesCount = 4;
		const int keyPointDimensions = 3;
		var tensor = output.Output0;
		var stride = tensor.Strides[1];
		foreach (var pose in _poseProcessor.Process(output))
		{
			var keyPointsBuilder = ImmutableArray.CreateBuilder<KeyPoint3D>(pose.KeyPoints.Length);
			for (byte keyPointIndex = 0; keyPointIndex < pose.KeyPoints.Length; keyPointIndex++)
			{
				var offset = keyPointIndex * keyPointDimensions + boundingCoordinatesCount + _metadata.ClassesNames.Length;
				var keyPointConfidence = tensor.Buffer.Span[(offset + 2) * stride + pose.Detection.Index];
				KeyPoint keyPoint = pose.KeyPoints[keyPointIndex];
				KeyPoint3D keyPoint3D = new(keyPoint.Position, keyPointConfidence);
				keyPointsBuilder.Add(keyPoint3D);
			}
			Pose3D pose3D = new(pose.Detection, keyPointsBuilder.MoveToImmutable());
			yield return pose3D;
		}
	}

	private readonly Metadata _metadata;
	private readonly BoundedOutputProcessor<Pose> _poseProcessor;
}