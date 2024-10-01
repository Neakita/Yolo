using System.Collections.Immutable;
using CommunityToolkit.Diagnostics;

namespace Yolo.OutputProcessing;

public sealed class V8PoseProcessor : BoundedOutputProcessor<Pose>
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

	public IEnumerable<Pose> Process(RawOutput output)
	{
		const int boundingCoordinates = 4;
		var tensor = output.Output0;
		var stride = tensor.Strides[1];
		foreach (var detection in _detectionProcessor.Process(output))
		{
			var keyPointsBuilder = ImmutableArray.CreateBuilder<KeyPoint>(_poserMetadata.KeyPointsCount);
			for (int keyPointIndex = 0; keyPointIndex < _poserMetadata.KeyPointsCount; keyPointIndex++)
			{
				var offset = keyPointIndex * _poserMetadata.KeyPointsDimensions + boundingCoordinates + _metadata.ClassesNames.Length;
				var pointX = (int)tensor.Buffer.Span[offset * stride + detection.Index];
				var pointY = (int)tensor.Buffer.Span[(offset + 1) * stride + detection.Index];
				Vector2<float> position = new(pointX, pointY);
				KeyPoint keyPoint = new(position);
				keyPointsBuilder.Add(keyPoint);
			}
			yield return new Pose(detection, keyPointsBuilder.MoveToImmutable());
		}
	}

	private readonly Metadata _metadata;
	private readonly PoserMetadata _poserMetadata;
	private readonly V8DetectionProcessor _detectionProcessor;
}